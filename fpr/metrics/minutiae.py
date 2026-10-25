"""Extract minutiae with MINDTCT and compare two minutiae sets in a region.

"Spurious" minutiae (in the repair, absent from the clean print) are
hallucinated ridge endings or bifurcations; "missed" ones are lost. Used by
fpr/validation/quick_check.py (script 08) and the demo app. NBIS: tools/install_nbis.md.
"""
import re
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

import config
from fpr.metrics.matching import nbis

# .min line: "  12 :  210, 345 :  7 :  0.42 : BIF : ..." -> x, y, direction (0-31), type.
MIN_LINE = re.compile(r"^\s*\d+\s*:\s*(\d+),\s*(\d+)\s*:\s*(\d+)\s*:\s*[\d.]+\s*:\s*(RIG|BIF)")
DIRECTION_UNIT_DEG = 11.25          # mindtct directions are in 32 steps of 11.25 deg


def read_min_file(path: Path) -> np.ndarray:
    """Parse a mindtct .min file into float (N, 4): x, y (pixels, top-left origin),
    theta (degrees, mindtct convention), type (0 = ridge ending, 1 = bifurcation)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        match = MIN_LINE.match(line)
        if match:
            x, y, direction, kind = match.groups()
            rows.append([float(x), float(y), int(direction) * DIRECTION_UNIT_DEG,
                         1.0 if kind == "BIF" else 0.0])
    return np.array(rows, dtype=float).reshape(-1, 4)


def extract_minutiae(image: np.ndarray) -> np.ndarray:
    """Minutiae of a uint8 (H, W) 500 ppi print as float (N, 4): x, y, theta_deg, type.

    Runs mindtct in a temporary folder.
    """
    with tempfile.TemporaryDirectory() as tmp:
        _, min_path = nbis.run_mindtct(image, Path(tmp) / "m")
        return read_min_file(min_path)


def inside(minutiae: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Bool (N,): which minutiae fall on a True pixel of `region` (bool (H, W))."""
    cols = np.clip(minutiae[:, 0].astype(int), 0, region.shape[1] - 1)
    rows = np.clip(minutiae[:, 1].astype(int), 0, region.shape[0] - 1)
    return region[rows, cols]


def match_minutiae(a: np.ndarray, b: np.ndarray, dist_px: float, angle_deg: float
                   ) -> list[tuple[int, int]]:
    """One-to-one matches (i in a, j in b) within dist_px and angle_deg (Hungarian)."""
    if len(a) == 0 or len(b) == 0:
        return []
    distance = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    turn = np.abs(a[:, None, 2] - b[None, :, 2]) % 360
    turn = np.minimum(turn, 360 - turn)
    allowed = (distance <= dist_px) & (turn <= angle_deg)
    cost = np.where(allowed, distance, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(rows, cols) if allowed[i, j]]


def compare_minutiae(recon: np.ndarray, clean: np.ndarray, region: np.ndarray,
                     dist_px: float = config.MINUTIA_MATCH_DIST_PX,
                     angle_deg: float = config.MINUTIA_MATCH_ANGLE_DEG) -> dict:
    """Compare minutiae (float (N, 4) arrays from extract_minutiae) inside a damaged region.

    The region is dilated by MINUTIA_REGION_DILATE_PX. Reconstructed minutiae in
    the region are matched one-to-one against all clean minutiae.
    Output: spurious_rate (unmatched recon / n_recon), missed_rate (unmatched
    clean / n_clean), n_clean, n_recon. Rates are NaN when the count is 0.
    """
    grown = ndimage.binary_dilation(region, iterations=config.MINUTIA_REGION_DILATE_PX)
    recon_in = recon[inside(recon, grown)]
    clean_in_mask = inside(clean, grown)
    matches = match_minutiae(recon_in, clean, dist_px, angle_deg)
    matched_clean = {j for _, j in matches}
    n_recon, n_clean = len(recon_in), int(clean_in_mask.sum())
    missed = sum(1 for j in np.flatnonzero(clean_in_mask) if j not in matched_clean)
    return {"spurious_rate": (n_recon - len(matches)) / n_recon if n_recon else float("nan"),
            "missed_rate": missed / n_clean if n_clean else float("nan"),
            "n_clean": n_clean, "n_recon": n_recon}
