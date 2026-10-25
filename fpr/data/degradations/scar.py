"""Scars and cuts: long, wide strokes with no ridges, and the ridges offset across them.

Scar tissue carries no ridges, so a scar prints as a ridge-free band (mid grey,
or bright when healed, or dark when the cut is fresh or dirt-filled). The skin on
the two sides heals slightly out of line, so ridges do not continue straight
across the scar. Registered in degradations/__init__.py.
"""
import numpy as np
from scipy import ndimage

import config
from fpr.data.degradations import shapes


def shear_field(shape: tuple[int, int], curve: np.ndarray, width: float,
                shear_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Displacement (d_row, d_col), float32 (H, W), that slides the two sides of a stroke apart.

    Each pixel moves along the stroke direction by +shear_px/2 on one side and
    -shear_px/2 on the other, fading to 0 at config.SCAR_SHEAR_REACH widths
    beyond the stroke edge. `curve` is the (N, 2) (row, col) centre line.
    """
    h, w = shape
    tangent = np.gradient(curve, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8
    rows = np.clip(np.round(curve[:, 0]).astype(int), 0, h - 1)
    cols = np.clip(np.round(curve[:, 1]).astype(int), 0, w - 1)
    centre = np.zeros(shape, bool)
    centre[rows, cols] = True
    t_row, t_col = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    t_row[rows, cols], t_col[rows, cols] = tangent[:, 0], tangent[:, 1]

    # Nearest centre-line pixel of every pixel, and which side of the line it is on.
    distance, (near_row, near_col) = ndimage.distance_transform_edt(~centre, return_indices=True)
    yy, xx = np.mgrid[0:h, 0:w]
    tr, tc = t_row[near_row, near_col], t_col[near_row, near_col]
    side = np.sign((yy - near_row) * tc - (xx - near_col) * tr)
    fade = np.clip(1 - (distance - width / 2) / (config.SCAR_SHEAR_REACH * width), 0, 1)
    amount = side * shear_px / 2 * fade
    return (amount * tr).astype(np.float32), (amount * tc).astype(np.float32)


def degrade(image: np.ndarray, fg_mask: np.ndarray, severity: float,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw 1-5 tapered Bezier strokes, ridge-free inside, with sheared ridges beside them.

    Input: uint8 (H, W) clean print, bool (H, W) foreground, severity in [0, 1].
    Output: (degraded uint8 (H, W), damage_mask bool (H, W)).
    Severity scales the number (3-5 at severity 1), length (-> 400 px), width
    (-> 20 px) and shear (2 -> 4 px) of the strokes.
    """
    low = int(round(shapes.lerp(config.SCAR_COUNT_MIN, severity)))
    high = int(round(shapes.lerp(config.SCAR_COUNT_MAX, severity)))
    stroke = np.zeros(image.shape, bool)
    d_row, d_col = np.zeros(image.shape, np.float32), np.zeros(image.shape, np.float32)
    for _ in range(rng.integers(low, high + 1)):
        length = shapes.lerp(config.SCAR_LENGTH_PX, severity) * rng.uniform(0.6, 1.0)
        curve = shapes.bezier_curve(shapes.random_foreground_point(fg_mask, rng), length, rng)
        max_width = shapes.lerp(config.SCAR_WIDTH_PX, severity) * rng.uniform(0.7, 1.0)
        # Width swells in the middle and tapers to the ends, like a real cut.
        widths = max_width * (0.4 + 0.6 * np.sin(np.linspace(0, np.pi, len(curve))))
        stroke |= shapes.draw_stroke(image.shape, curve, widths)
        shear = shapes.lerp(config.SCAR_SHEAR_PX, severity) * rng.uniform(0.8, 1.0)
        sr, sc = shear_field(image.shape, curve, max_width, shear * rng.choice([-1, 1]))
        d_row, d_col = d_row + sr, d_col + sc

    stroke_alpha = np.clip(ndimage.gaussian_filter(stroke.astype(np.float32), 1.0) * 1.5, 0, 1)
    texture = config.SCAR_TEXTURE * shapes.smooth_noise(image.shape, 1.5, rng)
    tones = list(config.SCAR_TONE_PROBS)
    tone = tones[rng.choice(len(tones), p=list(config.SCAR_TONE_PROBS.values()))]
    if tone == "flat":       # ridge-free skin: the print's local mean grey
        value = ndimage.gaussian_filter(image.astype(np.float32), config.SCAR_MEAN_SIGMA_PX)
    elif tone == "bright":   # healed scar
        value = np.full(image.shape, rng.uniform(*config.SCAR_BRIGHT_GREY), np.float32)
    else:                    # fresh cut or dirt-filled crease
        value = np.full(image.shape, rng.uniform(*config.SCAR_DARK_GREY), np.float32)

    sheared = shapes.warp(image, d_row, d_col)
    changed = (1 - stroke_alpha) * sheared + stroke_alpha * (value + texture)
    alpha = shapes.warp_cover(stroke_alpha, d_row, d_col)
    return shapes.finish(image, changed, alpha, fg_mask)
