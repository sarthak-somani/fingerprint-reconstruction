"""Score one repaired print exactly as evaluation/evaluate.py (eval version 2) scores it.

Every image is MSU-normalised first (clean probe, damaged probe, repair, gallery print), except a
repair from a model whose params.json says output_normalized (E12): that one is measured as stored,
as evaluate.py --rec-normalized does. The metric functions and the NBIS calls are imported from
evaluate.py itself (read-only), so the demo cannot drift from the evaluator:
  * match score: bozorth3 of the probe against the gallery print, MATCH when score > tau;
  * tau: the FMR-1% threshold on clean test impostors, read from the newest leaderboard row of this
    data version and eval version (23 on v2 test);
  * ridge fidelity: local NCC of band-passed ridges, averaged over the true damage mask;
  * PSNR in the true damage mask.
"""
import importlib.util
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config
from fpr import leaderboard
from fpr.data.normalization import normalize


@lru_cache(maxsize=1)
def evaluator():
    """evaluation/evaluate.py as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("evaluate", config.EVALUATE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def tau() -> float | None:
    """The evaluator's current tau for config.DEMO_SPLIT and config.DATA_VERSION (None if never evaluated)."""
    if not leaderboard.CSV.exists():
        return None
    rows = pd.read_csv(leaderboard.CSV)
    rows = rows[(rows["split"] == config.DEMO_SPLIT) & (rows["data_version"] == config.DATA_VERSION)
                & (rows["eval_version"] == leaderboard.EVAL_VERSION)]
    return float(rows["tau"].iloc[-1]) if len(rows) else None


def evaluator_view(image: np.ndarray) -> np.ndarray:
    """What the evaluator measures: the MSU-normalised print (the same sigma evaluate.py uses)."""
    return normalize(image, evaluator().NORMALIZE_SIGMA)


def ridge_fidelity(clean_n: np.ndarray, image_n: np.ndarray, region: np.ndarray) -> float:
    return evaluator().ridge_fidelity(clean_n, image_n, [region])[0]


def psnr(clean_n: np.ndarray, image_n: np.ndarray, region: np.ndarray) -> float:
    return evaluator().psnr(clean_n, image_n, region)


class Matcher:
    """bozorth3 scores of probes against one reference print; the reference's minutiae are extracted once."""

    def __init__(self, reference_n: np.ndarray):
        self.tmp = tempfile.TemporaryDirectory(prefix="demo_match_")
        self.reference_xyt = self.minutiae(reference_n, "reference")

    def minutiae(self, image_n: np.ndarray, name: str) -> Path:
        """mindtct on a normalised print, written as a 500 ppi PNG the way evaluate.py writes it."""
        ev, folder = evaluator(), Path(self.tmp.name)
        png = folder / f"{name}.png"
        Image.fromarray(image_n).save(png, dpi=(ev.STANDARD_PPI, ev.STANDARD_PPI))
        ev.run_mindtct(ev.find_tool("mindtct"), png, folder / name)
        return (folder / name).with_suffix(".xyt")

    def score(self, probe_n: np.ndarray, name: str) -> float:
        ev = evaluator()
        return float(ev.bozorth_scores(ev.find_tool("bozorth3"), self.minutiae(probe_n, name),
                                       [self.reference_xyt])[0])


def nbis_available() -> bool:
    ev = evaluator()
    return ev.find_tool("mindtct") is not None and ev.find_tool("bozorth3") is not None


def verdict(score: float | None) -> str:
    """'MATCH' / 'NO MATCH' at tau (accepted when score > tau, as in evaluate.py), or '' without tau."""
    if score is None or tau() is None:
        return ""
    return "MATCH" if score > tau() else "NO MATCH"
