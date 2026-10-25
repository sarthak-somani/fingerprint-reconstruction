"""Build genuine / impostor verification pairs for a split.

Mirrors Aadhaar: gallery = clean slap print (enrolment), probe = one plain
touch (damaged, then repaired). Used by preprocessing/06_make_pairs.py; the pairs
are scored by evaluation/evaluate.py and fpr/validation/.
"""
import pandas as pd


def build_pairs(probes: pd.DataFrame, gallery: pd.DataFrame) -> pd.DataFrame:
    """Every probe against every gallery slap print of the same split.

    Inputs: probes with probe_id, participant_id, finger; gallery with
    participant_id, finger, gallery_path.
    Output: DataFrame probe_id, gallery_path, label (1 = same participant and
    finger, 0 = anything else, including the same person's other fingers).
    """
    pairs = probes[["probe_id", "participant_id", "finger"]].merge(
        gallery, how="cross", suffixes=("", "_gallery"))
    same_person = pairs["participant_id"] == pairs["participant_id_gallery"]
    same_finger = pairs["finger"] == pairs["finger_gallery"]
    pairs["label"] = (same_person & same_finger).astype(int)
    return pairs[["probe_id", "gallery_path", "label"]].reset_index(drop=True)


def check_pairs(pairs: pd.DataFrame, probes: pd.DataFrame) -> None:
    """Assert exactly one genuine pair per probe and no duplicate pairs."""
    genuine_per_probe = pairs.groupby("probe_id")["label"].sum()
    assert (genuine_per_probe == 1).all(), "each probe needs exactly one genuine gallery print"
    assert set(genuine_per_probe.index) == set(probes["probe_id"])
    assert not pairs.duplicated(["probe_id", "gallery_path"]).any()
