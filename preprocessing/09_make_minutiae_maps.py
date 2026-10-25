"""Step 09: a minutiae weight map for every clean train crop -> preprocessing/data/train/minutiae_w/.

Usage: python preprocessing/09_make_minutiae_maps.py [--workers 12]
MINDTCT finds all minutiae of the clean crop; each one becomes a Gaussian bump (sigma 8 px, cut at
17 px), and the map is scaled to a peak of 1. Saved as float16 <clean file>.npy at crop size, plus
meta.csv. Reads clean crops only, so it does not depend on the damage version. Used by E06-E08
(FixedTrainDataset(minutiae_dir=...)). Files already done are skipped.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

import config
from fpr.data import image_io
from fpr.metrics import minutiae

META = config.TRAIN_MINUTIAE_DIR / "meta.csv"


def map_path(clean_file: str) -> Path:
    """Where the map of a pool.csv 'file' (e.g. K/<name>.png) is stored: minutiae_w/K/<name>.npy."""
    return config.TRAIN_MINUTIAE_DIR / Path(clean_file).with_suffix(".npy")


def minutiae_map(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """float32 (H, W) in [0, 1]: Gaussian bumps at the minutiae, peak 1; all zeros if there are none.

    points: float (N, 4) from fpr.metrics.minutiae.extract_minutiae (x, y first).
    """
    impulses = np.zeros(shape, np.float32)
    cols = np.clip(points[:, 0].astype(int), 0, shape[1] - 1)
    rows = np.clip(points[:, 1].astype(int), 0, shape[0] - 1)
    impulses[rows, cols] = 1.0
    sigma = config.MINUTIAE_MAP_SIGMA_PX
    bumps = ndimage.gaussian_filter(impulses, sigma, truncate=config.MINUTIAE_MAP_HALF_WINDOW_PX / sigma)
    return bumps / bumps.max() if bumps.max() > 0 else bumps


def map_one_crop(clean_file: str) -> dict:
    """Run MINDTCT on one clean crop, save its map and return its meta row."""
    clean = image_io.load_gray(config.TRAIN_CLEAN_DIR / clean_file)
    points = minutiae.extract_minutiae(clean)
    out = map_path(clean_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, minutiae_map(points, clean.shape).astype(np.float16))
    return {"file": clean_file, "n_minutiae": len(points), "shape": f"{clean.shape[0]}x{clean.shape[1]}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=config.MINUTIAE_MAP_WORKERS)
    args = parser.parse_args()

    pool = pd.read_csv(config.TRAIN_CLEAN_DIR / "pool.csv", dtype={"participant_id": str})
    # A crop is done when its .npy exists and meta.csv has its row (rows are appended as crops
    # finish, so an interrupted run resumes where it stopped).
    done = set()
    if META.exists():
        done = {f for f in pd.read_csv(META)["file"] if map_path(f).exists()}
    todo = [f for f in pool["file"] if f not in done]
    print(f"{len(pool)} crops, {len(done)} already done, {len(todo)} to do", flush=True)

    META.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as executor, META.open("a") as log:
        if log.tell() == 0:
            log.write("file,n_minutiae,shape\n")
        for row in tqdm(executor.map(map_one_crop, todo, chunksize=8), total=len(todo),
                        desc="minutiae maps", mininterval=30):
            log.write(f"{row['file']},{row['n_minutiae']},{row['shape']}\n")
            log.flush()
    # Rewrite in pool order, one row per crop (drops rows of maps that were deleted).
    meta = pd.read_csv(META).drop_duplicates("file", keep="last").set_index("file")
    meta = meta.loc[pool["file"]].reset_index()
    meta.to_csv(META, index=False)
    counts = meta["n_minutiae"]
    print(f"wrote {len(meta)} maps to {config.TRAIN_MINUTIAE_DIR}; minutiae per crop: "
          f"median {counts.median():.0f}, min {counts.min()}, max {counts.max()}, "
          f"zero in {(counts == 0).sum()} crops")


if __name__ == "__main__":
    main()
