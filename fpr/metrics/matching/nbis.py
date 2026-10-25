"""Thin wrappers around NIST NBIS: mindtct (minutiae) and bozorth3 (matching).

NIST Biometric Image Software (NBIS). Install: tools/install_nbis.md.
There is no fallback matcher: if the binaries are missing, require_nbis()
raises. Used by fpr/metrics/minutiae.py and fpr/validation/quick_check.py.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import config
from fpr.data import image_io


def missing_tools() -> list[str]:
    """Names of the NBIS programs that are not on PATH."""
    return [tool for tool in (config.MINDTCT, config.BOZORTH3) if shutil.which(tool) is None]


def require_nbis() -> None:
    """Raise RuntimeError, pointing at tools/install_nbis.md, if NBIS is not installed."""
    missing = missing_tools()
    if missing:
        raise RuntimeError(f"NBIS not found on PATH: {missing}. "
                           f"See {config.REPO_ROOT / 'tools' / 'install_nbis.md'}")


def run_mindtct(image: np.ndarray, out_root: Path) -> tuple[Path, Path]:
    """Run mindtct on a uint8 (H, W) 500 ppi image.

    Writes out_root.xyt (for bozorth3) and out_root.min (minutiae with type)
    and returns their paths. The image goes through a temporary PNG and all
    other mindtct outputs stay in the temp dir.
    """
    out_root = Path(out_root)
    out_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "image.png"
        image_io.save_png(png, image)
        subprocess.run([config.MINDTCT, str(png), str(Path(tmp) / "out")],
                       check=True, capture_output=True)
        for suffix in (".xyt", ".min"):
            shutil.copy(Path(tmp) / f"out{suffix}", out_root.with_suffix(suffix))
    return out_root.with_suffix(".xyt"), out_root.with_suffix(".min")


def cached_xyt(image_path: Path, cache_dir: Path) -> Path:
    """.xyt for an image file, computed once and kept in `cache_dir` (keyed by file stem)."""
    xyt = Path(cache_dir) / f"{Path(image_path).stem}.xyt"
    if not xyt.exists():
        run_mindtct(image_io.load_gray(image_path), xyt.with_suffix(""))
    return xyt


def bozorth_scores(probe_xyt: Path, gallery_xyts: list[Path]) -> np.ndarray:
    """bozorth3 score of one probe against each gallery file (higher = more similar).

    One process per BOZORTH_BATCH gallery files (`-p probe g1 g2 ...`),
    instead of one per pair. Output: float array in the order of `gallery_xyts`;
    a pair bozorth3 does not report gets score 0.
    """
    position = {str(path): k for k, path in enumerate(gallery_xyts)}
    scores = np.zeros(len(gallery_xyts))
    for start in range(0, len(gallery_xyts), config.BOZORTH_BATCH):
        batch = [str(p) for p in gallery_xyts[start:start + config.BOZORTH_BATCH]]
        result = subprocess.run([config.BOZORTH3, "-A", "outfmt=spg", "-p", str(probe_xyt)] + batch,
                                check=True, capture_output=True, text=True)
        for line in result.stdout.splitlines():
            fields = line.split()                 # score probe gallery
            if len(fields) == 3 and fields[2] in position:
                scores[position[fields[2]]] = float(fields[0])
    return scores


def nfiq2_score(image_path: Path) -> float | None:
    """NFIQ 2 quality (0-100) if the optional nfiq2 binary exists, else None.

    Parses the last integer printed by `nfiq2 <image>`. Unverified until nfiq2 is installed.
    """
    if shutil.which(config.NFIQ2) is None:
        return None
    result = subprocess.run([config.NFIQ2, str(image_path)], capture_output=True, text=True)
    numbers = [token for token in result.stdout.replace(",", " ").split() if token.isdigit()]
    return float(numbers[-1]) if numbers else None
