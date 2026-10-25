"""Synthetic contaminants: registry of {name: degrade function} and a random picker.

Every contaminant has the signature
    degrade(image uint8 HxW, fg_mask bool HxW, severity in [0, 1], rng)
        -> (degraded uint8 HxW, damage_mask bool HxW)
Used by fpr/data/datasets.py (on-the-fly training damage) and scripts 04-05.
"""
import numpy as np

import config
from fpr.data.degradations import lotion, mehendi, oil, pressure, sanitizer, scar

REGISTRY = {
    "pressure": pressure.degrade,
    "scar": scar.degrade,
    "oil": oil.degrade,
    "sanitizer": sanitizer.degrade,
    "mehendi": mehendi.degrade,
    "lotion": lotion.degrade,     # added last, so earlier round-robin orders keep their types
}
CONTAMINANTS = list(REGISTRY)


def apply_degradation(image: np.ndarray, fg_mask: np.ndarray, name: str, severity: float,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Apply one named contaminant. Same inputs/outputs as each degrade()."""
    assert image.dtype == np.uint8 and image.ndim == 2, "expected uint8 (H, W)"
    assert fg_mask.dtype == bool and fg_mask.shape == image.shape
    return REGISTRY[name](image, fg_mask, float(severity), rng)


def apply_random_degradation(image: np.ndarray, fg_mask: np.ndarray, rng: np.random.Generator,
                             names: list[str] | None = None
                             ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pick a contaminant (uniformly from `names`, default all) and a severity.

    Severity is uniform in config.SEVERITY_RANGE_TRAIN.
    Output: (degraded uint8 (H, W), damage_mask bool (H, W),
             info dict with 'contaminant' (str) and 'severity' (float)).
    """
    names = names or CONTAMINANTS
    name = names[rng.integers(len(names))]
    severity = rng.uniform(*config.SEVERITY_RANGE_TRAIN)
    degraded, mask = apply_degradation(image, fg_mask, name, severity, rng)
    return degraded, mask, {"contaminant": name, "severity": float(severity)}
