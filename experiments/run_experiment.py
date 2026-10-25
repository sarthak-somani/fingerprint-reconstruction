"""Repair every probe of a split with one experiment's method.

Usage: python experiments/run_experiment.py --exp E01_gabor --split test
Writes experiments/<group>/<exp>/reconstructions/{split}/<id>_<type>_rec.png (the names
evaluation/evaluate.py reads) and timing.csv (seconds per image, device).
Then evaluate with: python evaluation/run_eval.py --exp E01_gabor --split test
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import pandas as pd
from tqdm import tqdm

import config
from fpr import experiments
from fpr.data import eval_sets, image_io


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, choices=experiments.list_experiments())
    parser.add_argument("--split", required=True, choices=config.EVAL_SPLITS)
    parser.add_argument("--limit", type=int, default=None, help="only the first N probes (debug)")
    args = parser.parse_args()

    experiments.check_data_version(args.exp)
    meta = eval_sets.load_meta(args.split).head(args.limit)
    method = experiments.build_reconstructor(args.exp)
    device = getattr(method, "device", "cpu")
    out_dir = experiments.reconstructions_dir(args.exp, args.split)

    rows = []
    for row in tqdm(meta.itertuples(), total=len(meta), desc=f"{args.exp}/{args.split}", mininterval=10):
        degraded = image_io.load_gray(eval_sets.triplet_path(args.split, row.sample_id, "degraded"))
        mask = image_io.load_mask(eval_sets.triplet_path(args.split, row.sample_id, "mask"))
        start = time.perf_counter()
        repaired = method.repair(degraded, mask)
        rows.append({"sample_id": row.sample_id, "seconds": time.perf_counter() - start, "device": device})
        image_io.save_png(experiments.reconstruction_path(args.exp, args.split, row.probe_id,
                                                          row.contaminant), repaired)

    timing = pd.DataFrame(rows)
    timing.to_csv(out_dir / "timing.csv", index=False)
    print(f"wrote {len(timing)} reconstructions to {out_dir} "
          f"({timing['seconds'].mean():.2f} s/image on {device}, "
          f"{method.num_parameters():,} parameters)")


if __name__ == "__main__":
    main()
