"""Repair v3 test with one E27 run: a model's checkpoint + its fixed CCDF configuration + one sampler seed.

Usage:  FPR_DATA_VERSION=v3 python experiments/main/E27_seed_study/sample.py --run E22_s1_r2 [--limit N] [--root DIR]
Builds the CcdfReconstructor of E09 / E22 (fpr/models/reconstructors.py) on the run's checkpoint, with the
model's selected.json and inference.seed = the sampler seed (the reconstructor calls torch.manual_seed(seed)
before every print, so a run is reproducible and seed 0 is E09's / E22's own repair).
Writes reconstructions/v3/test/<run>/<id>_<type>_rec.png (the names evaluate.py reads), then timing.csv last.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: makes config/fpr importable

import pandas as pd
import torch

import config
from experiments.main.E09_ccdf.reconstructor import E02_PARAMS
from experiments.main.E27_seed_study import runs
from fpr.data import eval_sets, image_io
from fpr.models.reconstructors import CcdfReconstructor, numbers


def build(model: str, train_seed: int, sampler_seed: int, tile_batch: int = 64) -> CcdfReconstructor:
    """E09's sampler on the run's checkpoint, with the model's fixed configuration."""
    path = runs.checkpoint(model, train_seed)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    # The sampler builds E02's network (the diffusion settings are checked by the reconstructor).
    assert numbers(saved["params"]["model"]) == numbers(E02_PARAMS["model"]), f"{path}: network differs from E02's"
    assert saved["params"]["training"]["seed"] == train_seed, f"{path} was trained with another seed"
    inference = {"tile_batch": tile_batch, "seed": sampler_seed}
    return CcdfReconstructor(saved["params"], inference, path, runs.sampler_config(model))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--limit", type=int, default=None, help="only the first N samples (smoke test)")
    parser.add_argument("--root", type=Path, default=runs.HERE, help="output root (default this folder)")
    args = parser.parse_args()
    assert config.DATA_VERSION == runs.DATA_VERSION, "run with FPR_DATA_VERSION=v3"
    model, train_seed, sampler_seed = runs.parse(args.run)
    out = runs.recon_dir(args.run, args.root)
    out.mkdir(parents=True, exist_ok=True)
    method = build(model, train_seed, sampler_seed)
    meta = eval_sets.load_meta(runs.SPLIT).head(args.limit)
    print(f"{args.run}: {runs.checkpoint(model, train_seed)} (step {method.trained_steps}), "
          f"{json.dumps(method.config)}, sampler seed {sampler_seed}, {len(meta)} samples, {method.device}", flush=True)
    rows, start_all = [], time.time()
    for k, row in enumerate(meta.itertuples()):
        degraded = image_io.load_gray(eval_sets.triplet_path(runs.SPLIT, row.sample_id, "degraded"))
        mask = image_io.load_mask(eval_sets.triplet_path(runs.SPLIT, row.sample_id, "mask"))
        start = time.perf_counter()
        repaired = method.repair(degraded, mask)
        rows.append({"sample_id": row.sample_id, "seconds": time.perf_counter() - start, "device": method.device})
        image_io.save_png(out / eval_sets.layout_name(row.probe_id, row.contaminant, "reconstruction"), repaired)
        if (k + 1) % 200 == 0:
            print(f"{args.run}: {k + 1}/{len(meta)}  {time.time() - start_all:.0f} s", flush=True)
    timing = pd.DataFrame(rows)
    timing.to_csv(out / runs.TIME_FILE, index=False)
    print(f"{args.run}: wrote {len(timing)} repairs to {out} ({timing['seconds'].mean():.2f} s/image)", flush=True)


if __name__ == "__main__":
    main()
