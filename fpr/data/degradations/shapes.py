"""Shared helpers for the contaminants: blobs, lobes, Bezier strokes, warps, ridge dropout.

Used by every file in fpr/data/degradations/. All sizes are pixels at 500 ppi.
"""
import cv2
import numpy as np
from scipy import ndimage, special

import config


def lerp(low_high: tuple[float, float], t: float) -> float:
    """Linear interpolation between low_high[0] (t=0) and low_high[1] (t=1)."""
    return low_high[0] + (low_high[1] - low_high[0]) * t


def random_foreground_point(fg_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(row, col) float array of a random foreground pixel (any pixel if fg is empty)."""
    rows, cols = np.nonzero(fg_mask)
    if len(rows) == 0:
        return np.array([fg_mask.shape[0] / 2, fg_mask.shape[1] / 2])
    k = rng.integers(len(rows))
    return np.array([rows[k], cols[k]], dtype=float)


def disk(radius: int) -> np.ndarray:
    """Bool disk footprint of `radius` px (radius 0 = single pixel)."""
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return xx ** 2 + yy ** 2 <= radius ** 2


def smooth_noise(shape: tuple[int, int], sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian-smoothed white noise, float32 (H, W), rescaled to zero mean, unit std."""
    noise = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma)
    return (noise - noise.mean()) / (noise.std() + 1e-8)


def soft_blob(fg_mask: np.ndarray, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Organic blob around a random foreground point, as a blend weight.

    Output: float32 (H, W) in [0, 1]: 1 inside the blob, a soft ramp of width
    config.BLOB_EDGE * radius at the edge, 0 outside and off the finger.
    """
    h, w = fg_mask.shape
    centre = random_foreground_point(fg_mask, rng)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    distance = np.hypot(yy - centre[0], xx - centre[1]) / radius
    outline = distance + config.BLOB_ROUGHNESS * smooth_noise((h, w), config.BLOB_NOISE_SIGMA_PX, rng)
    alpha = np.clip((1.0 - outline) / config.BLOB_EDGE, 0.0, 1.0)
    return alpha * fg_mask


def gaussian_lobe(fg_mask: np.ndarray, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Elliptical Gaussian lobe around a random foreground point, as a blend weight.

    The weight is 1 at the centre and falls to config.MASK_ALPHA_MIN at `radius` px
    along the major axis (the minor axis is config.PRESSURE_LOBE_ASPECT times shorter).
    Output: float32 (H, W) in [0, 1], 0 off the finger.
    """
    h, w = fg_mask.shape
    centre = random_foreground_point(fg_mask, rng)
    angle = rng.uniform(0, np.pi)
    aspect = rng.uniform(*config.PRESSURE_LOBE_ASPECT)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    along = (yy - centre[0]) * np.sin(angle) + (xx - centre[1]) * np.cos(angle)
    across = (-(yy - centre[0]) * np.cos(angle) + (xx - centre[1]) * np.sin(angle)) / aspect
    distance_sq = (along ** 2 + across ** 2) / radius ** 2
    return np.exp(np.log(config.MASK_ALPHA_MIN) * distance_sq) * fg_mask


def bezier_curve(start: np.ndarray, length: float, rng: np.random.Generator,
                 n_points: int = 200, angle: float | None = None) -> np.ndarray:
    """Random cubic Bezier curve starting at `start` (row, col).

    The end point lies `length` px away at `angle` (radians; random if None); the
    two inner control points are pushed sideways by up to config.SCAR_BEND * length.
    Output: float (n_points, 2) array of (row, col) points.
    """
    if angle is None:
        angle = rng.uniform(0, 2 * np.pi)
    direction = np.array([np.sin(angle), np.cos(angle)])
    normal = np.array([-direction[1], direction[0]])
    end = start + length * direction
    bend = config.SCAR_BEND * length
    p1 = start + direction * length / 3 + normal * rng.uniform(-bend, bend)
    p2 = start + direction * 2 * length / 3 + normal * rng.uniform(-bend, bend)
    t = np.linspace(0, 1, n_points)[:, None]
    return ((1 - t) ** 3 * start + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * end)


def draw_stroke(shape: tuple[int, int], curve: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Rasterise a curve with per-point width into a bool (H, W) mask.

    `curve` is (N, 2) (row, col); `widths` is (N,) stroke width in px.
    Drawn as filled circles along the curve, so the width can vary smoothly.
    """
    canvas = np.zeros(shape, np.uint8)
    for (row, col), width in zip(curve, widths):
        cv2.circle(canvas, (int(round(col)), int(round(row))), max(1, int(round(width / 2))), 1, -1)
    return canvas.astype(bool)


def finish(clean: np.ndarray, degraded: np.ndarray, alpha: np.ndarray,
           fg_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Blend, clip to uint8, and make sure nothing changes outside the damage.

    Inputs: clean uint8 (H, W); degraded float (H, W) in grey levels; alpha
    float (H, W) in [0, 1] blend weight; fg_mask bool (H, W).
    Output: (degraded uint8 (H, W), damage_mask bool (H, W)). Pixels outside
    damage_mask are exactly equal to `clean`.
    """
    alpha = alpha * fg_mask
    blended = (1 - alpha) * clean.astype(np.float32) + alpha * degraded
    damage_mask = alpha > config.MASK_ALPHA_MIN
    out = np.where(damage_mask, np.clip(np.round(blended), 0, 255), clean).astype(np.uint8)
    return out, damage_mask


def smooth_displacement(shape: tuple[int, int], amplitude: float, sigma: float,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Smooth random displacement field (d_row, d_col), float32 (H, W) each, in px.

    Each component is smooth noise (Gaussian sigma `sigma` px, so a wavelength of
    about 4.4 * sigma) scaled so its 2-sigma value is `amplitude` px, clipped there.
    """
    d_row = amplitude * np.clip(smooth_noise(shape, sigma, rng) / 2, -1, 1)
    d_col = amplitude * np.clip(smooth_noise(shape, sigma, rng) / 2, -1, 1)
    return d_row, d_col


def warp(image: np.ndarray, d_row: np.ndarray, d_col: np.ndarray) -> np.ndarray:
    """Resample `image` at (row + d_row, col + d_col) with bilinear interpolation.

    Input: uint8 or float (H, W) image, float (H, W) displacements in px.
    Output: float32 (H, W) in grey levels.
    """
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coords = np.stack([yy + d_row, xx + d_col])
    return ndimage.map_coordinates(image.astype(np.float32), coords, order=1, mode="nearest")


def warp_cover(alpha: np.ndarray, d_row: np.ndarray, d_col: np.ndarray) -> np.ndarray:
    """Grow a blend weight so every displaced pixel lies inside the damage.

    A pixel moved by >= config.WARP_COVER_PX gets weight 1; smaller moves get a
    proportional weight. Output: float32 (H, W) in [0, 1], >= alpha everywhere.
    """
    moved = np.clip(np.hypot(d_row, d_col) / config.WARP_COVER_PX, 0.0, 1.0)
    return np.maximum(alpha, moved).astype(np.float32)


def valley_level(image: np.ndarray) -> np.ndarray:
    """Local background (valley) grey level, float32 (H, W): a smoothed max filter."""
    local_max = ndimage.maximum_filter(image.astype(np.float32), config.RIDGE_PERIOD_PX)
    return ndimage.gaussian_filter(local_max, config.RIDGE_PERIOD_PX / 3)


def ridge_dropout(image: np.ndarray, probability: np.ndarray | float,
                  rng: np.random.Generator, max_grey: float = config.DROPOUT_GREY_MAX) -> np.ndarray:
    """Break ridges into fragments: a random share of pixels takes the valley grey.

    Input: uint8 or float (H, W) image; `probability` = share of pixels dropped,
    a float or a float (H, W) map in [0, 1]. Gaps follow fine smooth noise, so they
    are short dashes rather than single pixels. A gap pixel becomes the local valley
    level, capped at `max_grey` (default config.DROPOUT_GREY_MAX) so damage never
    turns pure white.
    Output: float32 (H, W) in grey levels.
    """
    out = image.astype(np.float32)
    grain = special.ndtr(smooth_noise(image.shape, config.DROPOUT_SIGMA_PX, rng))  # uniform in [0, 1]
    gaps = grain < probability
    fill = np.minimum(valley_level(image), max_grey)
    out[gaps] = np.maximum(out[gaps], fill[gaps])     # a gap only ever lightens a ridge
    return out
