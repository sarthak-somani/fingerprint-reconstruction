"""Step 02: seeded participant-level split -> preprocessing/data/split.csv.

Usage: python preprocessing/02_make_split.py [--seed 643]
140 train / 20 val / 40 test people; val/test drawn from people with L + slap prints.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import numpy as np
import pandas as pd

import config
from fpr.data import data_md, nist, split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    args = parser.parse_args()

    index = nist.load_index()
    split_table = split.make_split(index, np.random.default_rng(args.seed))
    split_table.to_csv(config.SPLIT_CSV, index=False)

    merged = index.merge(split_table, on="participant_id")
    images = pd.crosstab(merged["split"], merged["device"] + "/" + merged["capture"])
    images.insert(0, "participants", split_table["split"].value_counts())
    ready = split.eval_ready_fingers(index).rename("fingers").reset_index()
    ready = ready.merge(split_table, on="participant_id")
    images["L+slap fingers"] = ready.groupby("split")["fingers"].sum()
    table = images.reindex(["train", "val", "test"]).reset_index()

    data_md.write_section("split", "## Participant split (script 02)\n\n"
                          + f"Seed {args.seed}. Val/test people need an L plain print and a "
                          + f"slap print for at least {config.MIN_EVAL_FINGERS} fingers.\n\n"
                          + data_md.table_to_markdown(table))
    print(table.to_string(index=False))
    print(f"wrote {config.SPLIT_CSV}")


if __name__ == "__main__":
    main()
