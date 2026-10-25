"""Lotion: creamy clumps that fill the valleys so neighbouring ridges bridge.

Unlike oil, lotion does not spread as one film: it sits in patchy clumps, and inside
each clump the valleys are filled and the ridges join up. Registered in
degradations/__init__.py.
"""
import numpy as np
from scipy import ndimage

import config
from fpr.data.degradations import shapes


def degrade(image: np.ndarray, fg_mask: np.ndarray, severity: float,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Threshold a smooth random field into clumps and close the ridges inside them.

    Input: uint8 (H, W) clean print, bool (H, W) foreground, severity in [0, 1].
    Output: (degraded uint8 (H, W), damage_mask bool (H, W)).
    Severity scales the clump coverage (10% -> 50% of the finger) and the closing
    radius (2 -> 4 px). Ridges are dark, so closing the ridges = grey-level opening.
    """
    field = shapes.smooth_noise(image.shape, config.LOTION_CLUMP_SIGMA_PX, rng)
    coverage = shapes.lerp(config.LOTION_COVERAGE, severity)
    threshold = np.quantile(field[fg_mask], 1 - coverage) if fg_mask.any() else np.inf
    # Soft edge inside the clump: the weight is MASK_ALPHA_MIN at the threshold, so the
    # damage mask is exactly the clumps and covers `coverage` of the finger.
    alpha = np.clip((field - threshold) / config.LOTION_EDGE + config.MASK_ALPHA_MIN, 0, 1) * fg_mask

    radius = int(round(shapes.lerp(config.LOTION_CLOSE_PX, severity)))
    filled = ndimage.grey_opening(image, footprint=shapes.disk(radius)).astype(np.float32)
    filled = ndimage.gaussian_filter(filled, config.MORPH_SMOOTH_SIGMA_PX)   # no round dot artefacts
    filled *= shapes.lerp(config.LOTION_DARKEN, severity)
    return shapes.finish(image, filled, alpha, fg_mask)
