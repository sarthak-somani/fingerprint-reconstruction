"""Step 10: MSU-normalised copies of the data, for models that work on normalised prints (E12).

Usage: FPR_DATA_VERSION=v3 python preprocessing/10_normalize.py [--workers 32] [--parts train val test]
Writes fpr.data.normalization.normalize() (texture of MSU's cartoon-texture decomposition, sigma
config.NORMALIZE_SIGMA) of:
- train: every clean crop -> train/clean_norm/ (shared by all versions) and every damaged sample of the
  active version -> config.TRAIN_DAMAGED_NORM_DIR/<sample>_degraded.png;
- val, test: every sample's clean and degraded print -> {split}/probes_norm/<sample>_{clean,degraded}.png.
Masks are not copied: normalisation does not move pixels, so the stored masks still apply.
Existing files are skipped, so an interrupted run can be restarted.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import pandas as pd
from tqdm import tqdm

import config
from fpr.data import data_md, eval_sets
from fpr.data.normalization import normalize_file


def train_jobs() -> list[tuple[Path, Path]]:
    pool = pd.read_csv(config.TRAIN_CLEAN_DIR / "pool.csv")
    meta = pd.read_csv(config.TRAIN_META_CSV)
    jobs = [(config.TRAIN_CLEAN_DIR / f, config.TRAIN_CLEAN_NORM_DIR / f) for f in pool["file"]]
    jobs += [(config.TRAIN_DAMAGED_DIR / f"{s}_degraded.png", config.TRAIN_DAMAGED_NORM_DIR / f"{s}_degraded.png")
             for s in meta["sample_id"]]
    return jobs


def split_jobs(split_name: str) -> list[tuple[Path, Path]]:
    meta = eval_sets.load_meta(split_name)
    return [(eval_sets.triplet_path(split_name, s, kind), eval_sets.normalized_triplet_path(split_name, s, kind))
            for s in meta["sample_id"] for kind in ("clean", "degraded")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=config.NORMALIZE_WORKERS)
    parser.add_argument("--parts", nargs="+", default=["val", "test", "train"], choices=["train", "val", "test"])
    args = parser.parse_args()

    rows = []
    for part in args.parts:
        jobs = train_jobs() if part == "train" else split_jobs(part)
        todo = [(src, dst) for src, dst in jobs if not dst.exists()]
        for folder in {dst.parent for _, dst in todo}:
            folder.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(args.workers) as executor:
            list(tqdm(executor.map(normalize_file, *zip(*todo), chunksize=16) if todo else [], total=len(todo),
                      desc=f"normalise {part}", mininterval=30))
        missing = sum(not dst.exists() for _, dst in jobs)
        assert missing == 0, f"{part}: {missing} normalised files missing"
        rows.append({"part": part, "images": len(jobs), "written now": len(todo)})
        print(f"{part}: {len(jobs)} normalised images ({len(todo)} written now)")
    data_md.write_section(data_md.versioned("normalized"),
                          f"## MSU-normalised copies, data {config.DATA_VERSION} (script 10)\n\n"
                          + f"Texture of MSU's cartoon-texture decomposition (σ {config.NORMALIZE_SIGMA:g}; "
                          + "`fpr/data/normalization.py`), for models that work on normalised prints: "
                          + "`train/clean_norm/`, `train/damaged_norm/` (degraded only), `{split}/probes_norm/` "
                          + "(clean and degraded). Masks are the stored ones.\n\n"
                          + data_md.table_to_markdown(pd.DataFrame(rows)))


if __name__ == "__main__":
    main()
