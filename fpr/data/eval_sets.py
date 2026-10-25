"""Which prints form the fixed val/test sets, and where their files live.

A probe is one finger of one val/test person that has both an L plain print
(damaged -> probe) and a slap print (clean gallery). Each split lives in
config.VERSION_DIR/{val,test}/: meta.csv, pairs.csv, probes/, probes_norm/, layout/.
A *sample* is one damaged copy of a probe: v2 damages each probe once (sample_id = probe_id),
v3 three times with three different contaminants (sample_id = '{probe_id}_{contaminant}').
Used by preprocessing scripts 05-07, experiments, evaluation and the demo app.
"""
from pathlib import Path

import pandas as pd

import config
from fpr.data import split


def probe_table(index: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """One row per probe finger of `split_name` ('val' or 'test').

    Output columns: probe_id ('{participant}_{finger:02d}'), participant_id,
    finger, probe_path (raw L print, relative to NIST_DIR), gallery_path
    (raw slap print, relative to NIST_DIR). Sorted by probe_id.
    """
    people = split.participants_in(split_name)
    in_split = index[index["participant_id"].isin(people)]
    plain = in_split[(in_split["capture"] == "plain") & (in_split["device"] == config.PROBE_DEVICE)]
    slap = in_split[in_split["capture"] == "slap"]
    probes = plain.merge(slap, on=["participant_id", "finger"], suffixes=("_probe", "_gallery"))
    probes = probes.rename(columns={"path_probe": "probe_path", "path_gallery": "gallery_path"})
    probes["probe_id"] = probes["participant_id"] + "_" + probes["finger"].map("{:02d}".format)
    columns = ["probe_id", "participant_id", "finger", "probe_path", "gallery_path"]
    return probes[columns].sort_values("probe_id").reset_index(drop=True)


def gallery_table(index: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """All slap prints of `split_name` people: participant_id, finger, gallery_path."""
    people = split.participants_in(split_name)
    slap = index[index["participant_id"].isin(people) & (index["capture"] == "slap")]
    slap = slap.rename(columns={"path": "gallery_path"})
    return slap[["participant_id", "finger", "gallery_path"]].sort_values("gallery_path")


def split_dir(split_name: str) -> Path:
    """config.VERSION_DIR/{split}/ for 'val' or 'test' (preprocessing/data/{split}/ for v2)."""
    assert split_name in config.EVAL_SPLITS, split_name
    return config.VERSION_DIR / split_name


def meta_path(split_name: str) -> Path:
    """preprocessing/data/{split}/meta.csv: one row per probe (written by script 05)."""
    return split_dir(split_name) / config.META_CSV


def pairs_path(split_name: str) -> Path:
    """preprocessing/data/{split}/pairs.csv: genuine / impostor pairs (written by script 06)."""
    return split_dir(split_name) / config.PAIRS_CSV


def triplet_path(split_name: str, sample_id: str, kind: str) -> Path:
    """{split}/probes/{sample_id}_{kind}.png, kind in clean/degraded/mask (sample_id = probe_id in v2)."""
    assert kind in {"clean", "degraded", "mask"}, kind
    return split_dir(split_name) / config.PROBES_SUBDIR / f"{sample_id}_{kind}.png"


def normalized_triplet_path(split_name: str, sample_id: str, kind: str) -> Path:
    """{split}/probes_norm/{sample_id}_{kind}.png: the MSU-normalised clean or degraded print (script 10)."""
    assert kind in {"clean", "degraded"}, kind
    return split_dir(split_name) / config.PROBES_NORM_SUBDIR / f"{sample_id}_{kind}.png"


def gallery_copy_path(split_name: str, gallery_path: str) -> Path:
    """preprocessing/data/{split}/gallery/<raw slap file name>: a copy of the raw slap print
    (written by script 07), so the split folder does not depend on the raw data."""
    return split_dir(split_name) / config.GALLERY_SUBDIR / Path(gallery_path).name


# ---------------------------------------------------------------- evaluation/evaluate.py naming
# The evaluator (evaluation/evaluate.py) reads four folders per split. Ids are our probe_ids
# ('00002312_01'), types are contaminant names.
LAYOUT_PARTS = ("original", "degraded", "mask", "gallery")


def layout_dir(split_name: str, part: str) -> Path:
    """preprocessing/data/{split}/layout/{part}/, part in LAYOUT_PARTS."""
    assert part in LAYOUT_PARTS, part
    return split_dir(split_name) / config.LAYOUT_SUBDIR / part


def layout_name(probe_id: str, contaminant: str, part: str) -> str:
    """File name evaluate.py expects: <id>.png, <id>_<type>.png, <id>_<type>_mask.png,
    <id>.png (gallery), or <id>_<type>_rec.png for part='reconstruction'."""
    names = {"original": f"{probe_id}.png", "gallery": f"{probe_id}.png",
             "degraded": f"{probe_id}_{contaminant}.png",
             "mask": f"{probe_id}_{contaminant}_mask.png",
             "reconstruction": f"{probe_id}_{contaminant}_rec.png"}
    return names[part]


def load_meta(split_name: str) -> pd.DataFrame:
    """Read {split}/meta.csv written by script 05: one row per sample.

    Sets sample_id = probe_id where the file has no sample_id column (v2 sets written before v3).
    """
    meta = pd.read_csv(meta_path(split_name),
                       dtype={"participant_id": str, "probe_id": str, "sample_id": str})
    if "sample_id" not in meta:
        meta.insert(0, "sample_id", meta["probe_id"])
    return meta
