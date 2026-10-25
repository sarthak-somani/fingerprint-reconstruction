"""Walk the raw NIST SD 302b/d folders and build one table of all images.

Used by preprocessing/01_index_dataset.py to write preprocessing/data/index.csv.
Data: Fiumara et al., NIST Technical Note 2007 (2019), SD 302.
"""
import re
from pathlib import Path

import cv2
import pandas as pd

import config

# SUBJECT_DEVICE_RESOLUTION_CAPTURE_FRGP.png, e.g. 00002437_K_500_plain_01.png
FILENAME_PATTERN = re.compile(
    r"^(?P<participant_id>\d{8})_(?P<device>[A-Z])_(?P<ppi>\d+)_"
    r"(?P<capture>[a-z]+)_(?P<finger>\d{2})\.png$"
)


def parse_filename(name: str) -> dict:
    """Split a NIST file name into its fields.

    Input: bare file name such as '00002437_K_500_plain_01.png'.
    Output: dict with participant_id (str, 8 digits), device (str),
    ppi (int), capture token (str, e.g. 'plain'/'roll'/'slap'), finger (int).
    Raises ValueError if the name does not follow the NIST pattern.
    """
    match = FILENAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Unexpected NIST file name: {name}")
    fields = match.groupdict()
    fields["ppi"] = int(fields["ppi"])
    fields["finger"] = int(fields["finger"])
    return fields


def index_folder(folder: Path, capture: str) -> list[dict]:
    """Return one row dict per PNG in `folder`.

    `capture` is the label stored in the table ('plain', 'rolled' or 'slap').
    Paths are stored relative to config.NIST_DIR so the table is portable.
    Height and width are read from the PNG header (pixels).
    """
    rows = []
    for png in sorted(folder.glob("*.png")):
        fields = parse_filename(png.name)
        image = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        rows.append({
            "path": str(png.relative_to(config.NIST_DIR)),
            "participant_id": fields["participant_id"],
            "finger": fields["finger"],
            "device": fields["device"],
            "capture": capture,
            "ppi": fields["ppi"],
            "height": image.shape[0],
            "width": image.shape[1],
        })
    return rows


def build_index() -> pd.DataFrame:
    """Index every folder in config.RAW_FOLDERS into one DataFrame.

    Columns: path, participant_id, finger (1-10), device, capture, ppi,
    height, width. Asserts 500 ppi only, finger codes 1-10, and that the
    count per (device, capture) equals config.EXPECTED_COUNTS.
    """
    rows = []
    for relative_folder, capture in config.RAW_FOLDERS:
        rows += index_folder(config.NIST_DIR / relative_folder, capture)
    index = pd.DataFrame(rows)
    check_index(index)
    return index


def check_index(index: pd.DataFrame) -> None:
    """Raise AssertionError if the index disagrees with the NIST documentation."""
    assert (index["ppi"] == config.PPI).all(), "found a non-500 ppi image"
    assert index["finger"].between(1, config.N_FINGERS).all(), "bad finger code"
    counts = index.groupby(["device", "capture"]).size().to_dict()
    assert counts == config.EXPECTED_COUNTS, f"counts {counts}"
    duplicated = index.duplicated(["participant_id", "device", "capture", "finger"])
    assert not duplicated.any(), "two images for the same finger and device"


def load_index() -> pd.DataFrame:
    """Read preprocessing/data/index.csv, keeping participant_id as an 8-digit string."""
    return pd.read_csv(config.INDEX_CSV, dtype={"participant_id": str})


def raw_path(relative_path: str) -> Path:
    """Turn an index.csv `path` entry into an absolute path to the raw PNG."""
    return config.NIST_DIR / relative_path
