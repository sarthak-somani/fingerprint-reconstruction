"""Where the segmenter's eval probes live, by data version.

v2 (current): preprocessing/data/{val,test}/. v1: kept in
preprocessing/data/v1_eval/{val,test}/ with the same probes/ and meta.csv (its layout/ links are stale).
"""
from pathlib import Path

import pandas as pd

import config

ROOTS = {"v1": config.DATA_DIR / "v1_eval", "v2": config.DATA_DIR}


def load_meta(version: str, split: str) -> pd.DataFrame:
    """The split's probe table (probe_id, contaminant, ...) for data `version`."""
    return pd.read_csv(ROOTS[version] / split / config.META_CSV, dtype={"probe_id": str, "participant_id": str})


def probe_path(version: str, split: str, probe_id: str, kind: str) -> Path:
    """<root>/<split>/probes/<probe_id>_<kind>.png, kind in {'clean', 'degraded', 'mask'}."""
    return ROOTS[version] / split / config.PROBES_SUBDIR / f"{probe_id}_{kind}.png"
