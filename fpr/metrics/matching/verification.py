"""Verification statistics from genuine and impostor score lists (pure numpy).

Higher score = more similar (bozorth3 convention). Used by scripts 08-09 and
the demo app (MATCH threshold).
"""
import numpy as np


def far_frr(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> tuple[float, float]:
    """(FAR, FRR) when 'match' means score >= threshold.

    FAR = fraction of impostor scores >= threshold; FRR = fraction of genuine < threshold.
    """
    return float(np.mean(impostor >= threshold)), float(np.mean(genuine < threshold))


def roc_points(genuine: np.ndarray, impostor: np.ndarray) -> dict[str, np.ndarray]:
    """FAR, FRR and TAR at every distinct score used as threshold.

    Inputs: 1-D float arrays of scores. Output: dict of 1-D arrays
    'threshold' (ascending), 'far', 'frr', 'tar' (= 1 - frr). A threshold
    above every score is appended, so far reaches 0.
    """
    genuine, impostor = np.asarray(genuine, float), np.asarray(impostor, float)
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    thresholds = np.append(thresholds, thresholds[-1] + 1)
    genuine_sorted, impostor_sorted = np.sort(genuine), np.sort(impostor)
    # Number of scores below each threshold, via binary search.
    frr = np.searchsorted(genuine_sorted, thresholds, side="left") / len(genuine)
    far = 1 - np.searchsorted(impostor_sorted, thresholds, side="left") / len(impostor)
    return {"threshold": thresholds, "far": far, "frr": frr, "tar": 1 - frr}


def eer(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Equal error rate in [0, 1]: where FAR and FRR cross.

    Taken at the threshold minimising |FAR - FRR|, reported as their mean.
    """
    points = roc_points(genuine, impostor)
    k = int(np.argmin(np.abs(points["far"] - points["frr"])))
    return float((points["far"][k] + points["frr"][k]) / 2)


def threshold_at_far(genuine: np.ndarray, impostor: np.ndarray, far: float) -> float:
    """Smallest threshold whose FAR is <= `far` (so the operating point is not optimistic)."""
    points = roc_points(genuine, impostor)
    allowed = points["far"] <= far
    return float(points["threshold"][np.argmax(allowed)])


def tar_at_far(genuine: np.ndarray, impostor: np.ndarray, far: float) -> float:
    """True accept rate at the smallest threshold with FAR <= `far`."""
    threshold = threshold_at_far(genuine, impostor, far)
    return float(np.mean(np.asarray(genuine) >= threshold))
