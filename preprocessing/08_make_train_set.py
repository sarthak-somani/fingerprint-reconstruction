"""Step 08: a fixed, stored damaged train set -> config.TRAIN_SET_DIR: damaged/ + meta.csv.

Usage: python preprocessing/08_make_train_set.py [--workers 16]
       FPR_DATA_VERSION=v3 python preprocessing/08_make_train_set.py   # -> preprocessing/data/v3/train/
Every clean train crop (train/clean/, from script 03) is damaged config.DAMAGE_COPIES times (v2: once,
v3: 3 times) with each of the contaminants (all six), at a severity drawn uniformly from
TRAIN_SET_SEVERITY_RANGE. v2 samples are '<crop>_<contaminant>', v3 '<crop>_<contaminant>_<copy>'.
Training can use these fixed files (FixedTrainDataset) or keep drawing fresh damage
from train/clean/ (TrainPatchDataset). Re-running gives identical files.
"""
import argparse
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from fpr.data import data_md, foreground, image_io
from fpr.data.degradations import apply_degradation, CONTAMINANTS


def damage_one_print(job: tuple[int, dict]) -> list[dict]:
    """config.DAMAGE_COPIES damaged versions per contaminant of one clean crop; returns their meta rows."""
    row_index, row = job
    clean = image_io.load_gray(config.TRAIN_CLEAN_DIR / row["file"])
    fg = foreground.foreground_mask(clean)
    rows = []
    for (k, name), copy in itertools.product(enumerate(CONTAMINANTS), range(config.DAMAGE_COPIES)):
        # One rng per (print, contaminant, copy): the same files whatever the worker count or order.
        # v2 (one copy) keeps its original seeds.
        key = [config.TRAIN_SET_SEED, row_index, k] + ([copy] if config.DAMAGE_COPIES > 1 else [])
        rng = np.random.default_rng(key)
        severity = float(rng.uniform(*config.TRAIN_SET_SEVERITY_RANGE))
        degraded, mask = apply_degradation(clean, fg, name, severity, rng)
        sample_id = f"{Path(row['file']).stem}_{name}" + (f"_{copy}" if config.DAMAGE_COPIES > 1 else "")
        image_io.save_png(config.TRAIN_DAMAGED_DIR / f"{sample_id}_degraded.png", degraded)
        image_io.save_png(config.TRAIN_DAMAGED_DIR / f"{sample_id}_mask.png", mask)
        rows.append({"sample_id": sample_id, "clean_file": row["file"],
                     "participant_id": row["participant_id"], "finger": row["finger"],
                     "device": row["device"], "capture": row["capture"], "contaminant": name,
                     "severity": round(severity, 3),
                     "damaged_fraction": round(float(mask.sum() / max(fg.sum(), 1)), 4)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=config.TRAIN_SET_WORKERS)
    args = parser.parse_args()

    pool = pd.read_csv(config.TRAIN_CLEAN_DIR / "pool.csv", dtype={"participant_id": str})
    jobs = list(enumerate(pool.to_dict("records")))
    config.TRAIN_DAMAGED_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as executor:
        results = list(tqdm(executor.map(damage_one_print, jobs, chunksize=16), total=len(jobs),
                            desc="train set", mininterval=10))
    meta = pd.DataFrame([r for rows in results for r in rows])
    meta.to_csv(config.TRAIN_META_CSV, index=False)

    summary = (meta.groupby("contaminant")
               .agg(samples=("sample_id", "size"),
                    mean_severity=("severity", lambda s: round(s.mean(), 2)),
                    mean_damaged_fraction=("damaged_fraction", lambda d: round(d.mean(), 3)))
               .reset_index())
    megabytes = sum(f.stat().st_size for f in config.TRAIN_DAMAGED_DIR.glob("*.png")) / 1e6
    times = "once" if config.DAMAGE_COPIES == 1 else f"{config.DAMAGE_COPIES} times"
    data_md.write_section(data_md.versioned("train_set"),
                          f"## Fixed damaged train set, data {config.DATA_VERSION} (script 08)\n\n"
                          + f"Each of the {len(pool)} clean train crops damaged {times} with each "
                          + f"contaminant (seed {config.TRAIN_SET_SEED}, severity uniform in "
                          + f"{config.TRAIN_SET_SEVERITY_RANGE}). {len(meta)} samples, "
                          + f"{megabytes:.0f} MB in `{config.TRAIN_DAMAGED_DIR.relative_to(config.DATA_DIR)}/`.\n\n"
                          + data_md.table_to_markdown(summary))
    print(summary.to_string(index=False))
    print(f"wrote {len(meta)} samples ({megabytes:.0f} MB) to {config.TRAIN_DAMAGED_DIR}")


if __name__ == "__main__":
    main()
