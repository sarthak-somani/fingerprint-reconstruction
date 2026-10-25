"""Step 00: link the raw datasets into preprocessing/raw_data/ (symlinks, nothing is copied).

Usage: python preprocessing/00_link_raw_data.py [--source ../data]
The raw data is checksum-verified and never moved; on another machine, point
--source at wherever NIST/ and SOCOFing/ live.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=config.RAW_DATA_SOURCE)
    args = parser.parse_args()
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name in config.RAW_DATASETS:
        target = (args.source / name).resolve()
        link = config.RAW_DATA_DIR / name
        if not target.is_dir():
            print(f"skip {name}: {target} not found")
            continue
        if link.is_symlink() or link.exists():
            assert link.resolve() == target, f"{link} already points to {link.resolve()}"
            print(f"ok   {link} -> {target}")
            continue
        link.symlink_to(target, target_is_directory=True)
        print(f"made {link} -> {target}")


if __name__ == "__main__":
    main()
