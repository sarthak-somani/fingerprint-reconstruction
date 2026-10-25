"""Damage-mask metrics: pixel confusion counts and IoU / Dice / precision / recall from them.

Counts are pooled (summed over probes) before scoring, so large and small masks weigh by their
pixels; per-probe Dice is reported separately by evaluate_zero_shot.py. Numpy only.
"""
import numpy as np

EPS = 1e-9


def confusion(pred: np.ndarray, truth: np.ndarray, region: np.ndarray | None = None) -> dict:
    """bool (H, W) predicted and true damage -> {'tp', 'fp', 'fn', 'tn'} pixel counts (inside `region`)."""
    region = np.ones(truth.shape, bool) if region is None else region
    p, t = pred[region], truth[region]
    return {"tp": int((p & t).sum()), "fp": int((p & ~t).sum()),
            "fn": int((~p & t).sum()), "tn": int((~p & ~t).sum())}


def scores(c: dict) -> dict:
    """IoU, Dice, precision, recall from confusion counts (pooled or for one probe)."""
    return {"iou": c["tp"] / (c["tp"] + c["fp"] + c["fn"] + EPS),
            "dice": 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"] + EPS),
            "precision": c["tp"] / (c["tp"] + c["fp"] + EPS),
            "recall": c["tp"] / (c["tp"] + c["fn"] + EPS)}
