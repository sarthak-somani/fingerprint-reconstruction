"""Step 03: export train-split prints, cropped to the finger, to preprocessing/data/train/clean/.

Usage: python preprocessing/03_export_training_pool.py
Includes 302d plain (K, L, M, P) and 302b rolled (U, V) of train people only.
This folder (+ pool.csv) is what TrainPatchDataset reads for GPU training on the Prajna cluster.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import pandas as pd
from tqdm import tqdm

import config
from fpr.data import data_md, foreground, image_io, nist, split


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    index = nist.load_index()
    train_people = set(split.participants_in("train"))
    chosen = index[index["participant_id"].isin(train_people)
                   & index["capture"].isin(["plain", "rolled"])]

    rows = []
    for row in tqdm(chosen.itertuples(), total=len(chosen), desc="pool"):
        image = image_io.load_gray(nist.raw_path(row.path))
        fg = foreground.foreground_mask(image, rolled=(row.capture == "rolled"))
        box = image_io.bounding_box(fg, margin=config.CROP_MARGIN_PX)
        cropped = image_io.crop(image, box)
        # Outside the finger becomes plain white: drops the joint strip and scanner borders.
        cropped[~image_io.crop(fg, box)] = config.BACKGROUND_VALUE
        file = f"{row.device}/{Path(row.path).name}"
        image_io.save_png(config.TRAIN_CLEAN_DIR / file, cropped)
        rows.append({"file": file, "participant_id": row.participant_id, "finger": row.finger,
                     "device": row.device, "capture": row.capture,
                     "height": cropped.shape[0], "width": cropped.shape[1],
                     "fg_fraction": round(float(image_io.crop(fg, box).mean()), 3)})

    pool = pd.DataFrame(rows)
    pool.to_csv(config.TRAIN_CLEAN_DIR / "pool.csv", index=False)
    summary = (pool.groupby(["device", "capture"])
               .agg(images=("file", "size"), median_h=("height", "median"),
                    median_w=("width", "median")).reset_index())
    megabytes = sum(f.stat().st_size for f in config.TRAIN_CLEAN_DIR.rglob("*.png")) / 1e6
    data_md.write_section("pool", "## Training pool (script 03)\n\n"
                          + f"Train-split prints cropped to the foreground box "
                          + f"(+{config.CROP_MARGIN_PX} px), background set to white. "
                          + f"{len(pool)} images, {megabytes:.0f} MB.\n\n"
                          + data_md.table_to_markdown(summary))
    print(summary.to_string(index=False))
    print(f"wrote {len(pool)} images ({megabytes:.0f} MB) to {config.TRAIN_CLEAN_DIR}")


if __name__ == "__main__":
    main()
