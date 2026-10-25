"""Mehendi (henna): a dyed design over the ridges, optionally with dark vine lines.

The henna design (here a plus sign or crossing strokes) dyes the skin and leaves
paste in the valleys: inside it ridge contrast drops, ridges break up, and the
dried paste changes the local ridge spacing. Fresh paste lines print as dark vines.
Registered in degradations/__init__.py.
"""
import numpy as np
from scipy import ndimage

import config
from fpr.data.degradations import shapes


def vine_mask(shape: tuple[int, int], fg_mask: np.ndarray, severity: float,
              rng: np.random.Generator) -> np.ndarray:
    """Bool (H, W) mask of 1-4 curved vines with small round 'leaves' along them."""
    vines = np.zeros(shape, bool)
    n_vines = int(round(shapes.lerp(config.MEHENDI_VINES, severity * rng.random())))
    for _ in range(max(1, n_vines)):
        length = shapes.lerp(config.MEHENDI_VINE_LENGTH_PX, severity) * rng.uniform(0.7, 1.2)
        curve = shapes.bezier_curve(shapes.random_foreground_point(fg_mask, rng), length, rng)
        width = shapes.lerp(config.MEHENDI_WIDTH_PX, severity) * rng.uniform(0.6, 1.0)
        widths = np.full(len(curve), width)
        # Leaves: every ~25 points the stroke briefly swells to 2-3x its width.
        leaf = (np.arange(len(curve)) % 25) < 4
        widths[leaf] *= rng.uniform(2.0, 3.0)
        vines |= shapes.draw_stroke(shape, curve, widths)
    return vines


def design_mask(shape: tuple[int, int], fg_mask: np.ndarray, severity: float,
                rng: np.random.Generator) -> np.ndarray:
    """Bool (H, W) mask of a plus sign or 2-3 crossing tapered strokes on the finger.

    The design is centred half-way between a random foreground point and the
    finger's centre, so it lands on the fingertip. Severity scales size and width.
    """
    rows, cols = np.nonzero(fg_mask)
    middle = np.array([rows.mean(), cols.mean()]) if len(rows) else np.array(shape) / 2
    centre = (shapes.random_foreground_point(fg_mask, rng) + middle) / 2
    size = shapes.lerp(config.MEHENDI_DESIGN_SIZE_PX, severity) * rng.uniform(0.8, 1.0)
    width = shapes.lerp(config.MEHENDI_DESIGN_WIDTH_PX, severity) * rng.uniform(0.8, 1.0)
    angle = rng.uniform(0, np.pi)
    design = np.zeros(shape, bool)
    if rng.random() < config.MEHENDI_PLUS_PROB:   # plus sign: two straight bars
        for bar_angle in (angle, angle + np.pi / 2):
            direction = np.array([np.sin(bar_angle), np.cos(bar_angle)])
            line = centre + np.linspace(-size / 2, size / 2, 200)[:, None] * direction
            design |= shapes.draw_stroke(shape, line, np.full(len(line), width))
    else:                                         # crossing tapered Bezier strokes
        n_strokes = rng.integers(config.MEHENDI_STROKES[0], config.MEHENDI_STROKES[1] + 1)
        for k in range(n_strokes):
            stroke_angle = angle + np.pi * k / n_strokes
            start = centre - size / 2 * np.array([np.sin(stroke_angle), np.cos(stroke_angle)])
            curve = shapes.bezier_curve(start, size, rng, angle=stroke_angle)
            widths = width * (0.4 + 0.6 * np.sin(np.linspace(0, np.pi, len(curve))))
            design |= shapes.draw_stroke(shape, curve, widths)
    return design & fg_mask


def degrade(image: np.ndarray, fg_mask: np.ndarray, severity: float,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Inside a henna design: warp, lower contrast, darken and break the ridges.

    Input: uint8 (H, W) clean print, bool (H, W) foreground, severity in [0, 1].
    Output: (degraded uint8 (H, W), damage_mask bool (H, W)).
    Severity scales the design size, warp (2 -> 4 px), contrast loss (50 -> 70%),
    dropout (30 -> 60%) and darkening; dark vines are added with
    probability config.MEHENDI_VINE_PROB.
    """
    design = design_mask(image.shape, fg_mask, severity, rng)
    alpha = np.clip(ndimage.gaussian_filter(design.astype(np.float32), 2.0) * 1.5, 0, 1)

    # Dried paste pushes ridges apart or together: a smooth warp, fading out with the design.
    d_row, d_col = shapes.smooth_displacement(image.shape, shapes.lerp(config.MEHENDI_WARP_PX, severity),
                                              config.MEHENDI_WARP_SIGMA_PX, rng)
    d_row, d_col = d_row * alpha, d_col * alpha
    changed = shapes.warp(image, d_row, d_col)

    # Broken ridges first (gaps take the print's own valley grey), then the contrast
    # drop towards the local mean and the dye's darkening, so faint ridges remain.
    changed = shapes.ridge_dropout(changed, shapes.lerp(config.MEHENDI_DROPOUT, severity), rng)
    local_mean = ndimage.gaussian_filter(changed, config.RIDGE_PERIOD_PX)
    changed = local_mean + (changed - local_mean) * (1 - shapes.lerp(config.MEHENDI_CONTRAST_DROP, severity))
    changed *= 1 - shapes.lerp(config.MEHENDI_STAIN_DARKEN, severity)
    alpha = shapes.warp_cover(alpha, d_row, d_col)

    if rng.random() < config.MEHENDI_VINE_PROB:
        vines = ndimage.gaussian_filter(vine_mask(image.shape, fg_mask, severity, rng).astype(np.float32), 0.8)
        vines = np.clip(vines * 1.5, 0, 1)
        ink = config.MEHENDI_INK + 10 * shapes.smooth_noise(image.shape, 2.0, rng)
        changed = (1 - vines) * changed + vines * ink
        alpha = np.maximum(alpha, vines)
    return shapes.finish(image, changed, alpha, fg_mask)
