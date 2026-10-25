"""Oil: a film over most of the fingertip that broadens and merges the ridges.

On an optical scanner the oil wets the valleys, so dark ridges spread and
neighbouring ridges merge into smears. Viscous (thick) oil does this strongly;
thin oil only slightly. Registered in degradations/__init__.py.
"""
import numpy as np
from scipy import ndimage

import config
from fpr.data.degradations import shapes


def degrade(image: np.ndarray, fg_mask: np.ndarray, severity: float,
            rng: np.random.Generator, viscous: bool | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
    """Broaden ridges (grey erosion) and darken them inside 1-3 large noisy blobs.

    Input: uint8 (H, W) clean print, bool (H, W) foreground, severity in [0, 1],
    `viscous` = strong (True) or mild (False) oil; drawn from rng if None.
    Output: (degraded uint8 (H, W), damage_mask bool (H, W)).
    Severity scales blob size and count, and for viscous oil the erosion radius
    (1 -> 3 px) and darkening. Ridges are dark, so broadening = grey erosion (min filter).
    """
    if viscous is None:
        viscous = bool(rng.random() < config.OIL_VISCOUS_PROB)
    alpha = np.zeros(image.shape, np.float32)
    n_blobs = config.OIL_BLOBS[0] + int(rng.random() < severity) + int(rng.random() < severity / 2)
    for _ in range(min(n_blobs, config.OIL_BLOBS[1])):
        radius = shapes.lerp(config.OIL_BLOB_RADIUS_PX, severity) * rng.uniform(0.8, 1.0)
        alpha = np.maximum(alpha, shapes.soft_blob(fg_mask, radius, rng))

    if viscous:
        radius = int(round(shapes.lerp(config.OIL_VISCOUS_RADIUS_PX, severity)))
        darken = shapes.lerp(config.OIL_DARKEN_VISCOUS, severity)
    else:
        radius, darken = config.OIL_THIN_RADIUS_PX, config.OIL_DARKEN_THIN
    smeared = ndimage.grey_erosion(image, footprint=shapes.disk(radius)).astype(np.float32)
    smeared = ndimage.gaussian_filter(smeared, config.MORPH_SMOOTH_SIGMA_PX)   # no round dot artefacts
    # Uneven film thickness: both the darkening and the blend weight vary smoothly.
    film = 0.5 + 0.5 * np.clip(shapes.smooth_noise(image.shape, config.OIL_FILM_SIGMA_PX, rng), -1, 1)
    smeared *= 1 - (1 - darken) * film
    alpha *= config.OIL_FILM_MIN + (1 - config.OIL_FILM_MIN) * film
    return shapes.finish(image, smeared, alpha, fg_mask)
