"""Uneven pressure: hard or light contact under 1-2 large Gaussian lobes.

Pressing hard squashes the skin flat: ridges spread into the valleys and bridge,
the print darkens and the ridge pattern shifts slightly. Pressing lightly leaves only
the ridge tops, which break up where contact fades towards the lobe edge.
Registered in degradations/__init__.py.
"""
import numpy as np
from scipy import ndimage

import config
from fpr.data.degradations import shapes


def degrade(image: np.ndarray, fg_mask: np.ndarray, severity: float,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Hard press (closing + darkening + warp) or light press (thinning + dropout).

    Input: uint8 (H, W) clean print, bool (H, W) foreground, severity in [0, 1].
    Output: (degraded uint8 (H, W), damage_mask bool (H, W)).
    Ridges are dark, so closing the ridges = grey-level opening of the image, and
    thinning = grey-level dilation (max filter).
    """
    alpha = np.zeros(image.shape, np.float32)
    n_lobes = config.PRESSURE_LOBES[0] + int(rng.random() < severity)
    for _ in range(min(n_lobes, config.PRESSURE_LOBES[1])):
        radius = shapes.lerp(config.PRESSURE_LOBE_RADIUS_PX, severity) * rng.uniform(0.8, 1.0)
        alpha = np.maximum(alpha, shapes.gaussian_lobe(fg_mask, radius, rng))

    if rng.random() < config.PRESSURE_HARD_PROB:   # hard press: ridges bridge, darker, squashed
        radius = int(round(shapes.lerp(config.PRESSURE_CLOSE_PX, severity)))
        closed = ndimage.grey_opening(image, footprint=shapes.disk(radius)).astype(np.float32)
        closed = ndimage.gaussian_filter(closed, config.MORPH_SMOOTH_SIGMA_PX)
        closed *= shapes.lerp(config.PRESSURE_DARKEN, severity)
        d_row, d_col = shapes.smooth_displacement(
            image.shape, shapes.lerp(config.PRESSURE_WARP_PX, severity),
            config.PRESSURE_WARP_SIGMA_PX, rng)
        d_row, d_col = d_row * alpha, d_col * alpha          # the warp fades out with the lobe
        changed = shapes.warp(closed, d_row, d_col)
        alpha = shapes.warp_cover(alpha, d_row, d_col)
    else:                                          # light press: thin, faint, broken towards the edge
        radius = int(round(shapes.lerp(config.PRESSURE_THIN_PX, severity)))
        thin = ndimage.grey_dilation(image, footprint=shapes.disk(radius)).astype(np.float32)
        thin = 255 - (255 - thin) * shapes.lerp(config.PRESSURE_LIGHT_FADE, severity)
        edge = np.clip((1 - alpha) / (1 - config.MASK_ALPHA_MIN), 0, 1)   # 0 at centre, 1 at edge
        changed = shapes.ridge_dropout(thin, shapes.lerp(config.PRESSURE_DROPOUT, severity) * edge, rng)
    return shapes.finish(image, changed, alpha, fg_mask)
