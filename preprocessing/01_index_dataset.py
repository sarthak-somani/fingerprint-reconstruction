"""Step 01: index every 500 ppi NIST image into preprocessing/data/index.csv.

Usage: python preprocessing/01_index_dataset.py
Asserts the per-scanner counts from the NIST documentation.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import config
from fpr.data import data_md, nist


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    index = nist.build_index()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    index.to_csv(config.INDEX_CSV, index=False)

    counts = (index.groupby(["device", "capture"])
              .agg(images=("path", "size"), participants=("participant_id", "nunique"),
                   height_range=("height", lambda h: f"{h.min()}-{h.max()}"),
                   width_range=("width", lambda w: f"{w.min()}-{w.max()}"))
              .reset_index())
    data_md.write_section("index", "## Image index (script 01)\n\n"
                          + data_md.table_to_markdown(counts)
                          + f"\n\nTotal: {len(index)} images at {config.PPI} ppi.")
    print(counts.to_string(index=False))
    print(f"wrote {config.INDEX_CSV} ({len(index)} images)")


if __name__ == "__main__":
    main()
