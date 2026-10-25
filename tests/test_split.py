"""Tests for the participant-level split (fpr/data/split.py, preprocessing/data/split.csv).

Checks that no participant is in two splits, sizes are 140/20/40, val/test
people have probe + gallery prints, and that split membership is only ever
read from split.csv.
"""
from pathlib import Path

import pytest

import config
from fpr.data import nist, split

needs_split = pytest.mark.skipif(not config.SPLIT_CSV.exists(), reason="run scripts 01-02 first")


@needs_split
def test_no_participant_in_two_splits():
    table = split.load_split()
    assert table["participant_id"].is_unique
    groups = [set(split.participants_in(s)) for s in ("train", "val", "test")]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])


@needs_split
def test_split_sizes_and_coverage():
    table = split.load_split()
    split.check_split(table)
    assert set(table["participant_id"]) == set(nist.load_index()["participant_id"])


@needs_split
def test_eval_people_have_probe_and_gallery():
    counts = split.eval_ready_fingers(nist.load_index())
    for name in ("val", "test"):
        assert (counts[split.participants_in(name)] >= config.MIN_EVAL_FINGERS).all()


def test_split_csv_only_read_through_split_module():
    """Every loader must get split membership from fpr.data.split (which reads split.csv)."""
    repo = config.REPO_ROOT
    allowed = {repo / "config.py", repo / "fpr" / "data" / "split.py", Path(__file__).resolve(),
               repo / "preprocessing" / "02_make_split.py"}          # the one writer
    # Our code only: evaluation/evaluate.py never touches the split.
    ours = (list((repo / "fpr").rglob("*.py")) + list((repo / "preprocessing").rglob("*.py"))
            + list((repo / "experiments").rglob("*.py")) + list((repo / "demo").glob("*.py"))
            + [repo / "evaluation" / "run_eval.py", repo / "evaluation" / "compare_experiments.py"])
    for source in ours:
        text = source.read_text()
        if source in allowed:
            continue
        assert "SPLIT_CSV" not in text and "split.csv" not in text.replace("preprocessing/data/split.csv", ""), \
            f"{source.name} reads split.csv directly; use fpr.data.split.participants_in()"
        if "participant" in text and ("'train'" in text or '"train"' in text
                                      or "'test'" in text or '"test"' in text):
            assert "participants_in" in text or "load_split" in text, \
                f"{source.name} selects by split without reading split.csv"
