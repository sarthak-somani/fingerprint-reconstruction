"""E17 val sweep: FingerGAN once per input kind on every v3 val sample, every output variant evaluated.

Usage (GPU):  FPR_DATA_VERSION=v3 python experiments/baselines/E17_fingergan_norm/sweep.py
Variants come from params.json "sweep"; the chosen one is written to selected.json (read by the
reconstructor). Results: sweep_val.csv, evaluation/results/E17_fingergan_norm/val_sweep_<label>_<time>/.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: makes config/fpr importable

import numpy as np

from experiments.baselines.E17_fingergan_norm import val_sweep
from experiments.baselines.E17_fingergan_norm.reconstructor import FingerGanNormReconstructor
from fpr import experiments

EXP = "E17_fingergan_norm"
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only write the first N samples (smoke test)")
    args = parser.parse_args()
    experiments.check_data_version(EXP)
    params = experiments.load_params(EXP)
    variants = params["sweep"]["variants"]
    method = FingerGanNormReconstructor(params, selected=params["inference"]["default"])

    def variants_fn(degraded: np.ndarray, mask: np.ndarray) -> dict:
        maps = {kind: method.ridge_map(degraded, kind) for kind in {v["input"] for v in variants.values()}}
        return {label: method.finish(maps[v["input"]], degraded, mask, v["render_sigma"], v["paste_back"])
                for label, v in variants.items()}

    if args.limit:                                   # smoke: write a few samples, no evaluation
        val_sweep.write_variants(EXP, "val", variants_fn, list(variants), limit=args.limit)
        return
    table = val_sweep.sweep(EXP, variants_fn, list(variants))
    best = val_sweep.choose(table)
    (HERE / "selected.json").write_text(json.dumps({"label": best, **variants[best]}, indent=2) + "\n")
    print(f"selected {best}: {variants[best]}")


if __name__ == "__main__":
    main()
