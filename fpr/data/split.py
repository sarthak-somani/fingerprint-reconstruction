"""Participant-level train/val/test split.

The split is by participant, never by image. preprocessing/02_make_split.py writes
preprocessing/data/split.csv; every other module gets split membership from
`load_split()` / `participants_in()` here, which read that file.
"""
import numpy as np
import pandas as pd

import config


def eval_ready_fingers(index: pd.DataFrame) -> pd.Series:
    """Count, per participant, fingers that have both an L plain print and a slap.

    Input: the index table from fpr.data.nist.build_index().
    Output: Series indexed by participant_id with an int count (0-10).
    """
    plain = index[(index["capture"] == "plain") & (index["device"] == config.PROBE_DEVICE)]
    slap = index[index["capture"] == "slap"]
    both = plain.merge(slap, on=["participant_id", "finger"])
    counts = both.groupby("participant_id")["finger"].nunique()
    everyone = index["participant_id"].unique()
    return counts.reindex(everyone, fill_value=0)


def make_split(index: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Assign every participant to 'train', 'val' or 'test'.

    Val and test people are drawn at random from those with at least
    config.MIN_EVAL_FINGERS L+slap fingers; everyone else is train.
    Output: DataFrame with columns participant_id, split (sorted by id).
    """
    counts = eval_ready_fingers(index)
    eligible = np.sort(counts[counts >= config.MIN_EVAL_FINGERS].index.to_numpy())
    n_eval = config.N_VAL + config.N_TEST
    assert len(eligible) >= n_eval, f"only {len(eligible)} eligible people"

    chosen = rng.permutation(eligible)[:n_eval]
    val_people = set(chosen[:config.N_VAL])
    test_people = set(chosen[config.N_VAL:])

    rows = []
    for participant in sorted(counts.index):
        if participant in val_people:
            split = "val"
        elif participant in test_people:
            split = "test"
        else:
            split = "train"
        rows.append({"participant_id": participant, "split": split})
    split_table = pd.DataFrame(rows)
    check_split(split_table)
    return split_table


def check_split(split_table: pd.DataFrame) -> None:
    """Raise AssertionError unless sizes are 140/20/40 and each person is in one split."""
    assert split_table["participant_id"].is_unique, "participant in two splits"
    sizes = split_table["split"].value_counts().to_dict()
    expected = {"train": config.N_TRAIN, "val": config.N_VAL, "test": config.N_TEST}
    assert sizes == expected, f"split sizes {sizes}"


def split_available() -> bool:
    """True if preprocessing/data/split.csv exists (it may be absent on a GPU node that only has the pool)."""
    return config.SPLIT_CSV.exists()


def load_split() -> pd.DataFrame:
    """Read preprocessing/data/split.csv (columns participant_id as 8-digit str, split)."""
    return pd.read_csv(config.SPLIT_CSV, dtype={"participant_id": str})


def participants_in(split: str) -> list[str]:
    """Sorted participant IDs belonging to `split` ('train', 'val' or 'test')."""
    split_table = load_split()
    assert split in {"train", "val", "test"}, f"unknown split {split}"
    return sorted(split_table.loc[split_table["split"] == split, "participant_id"])
