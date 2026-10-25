"""Ridge orientation field, filling it into damaged regions, and angular error.

Gradient method of Hong, Wan and Jain, "Fingerprint Image Enhancement:
Algorithm and Performance Evaluation", IEEE TPAMI 1998, computed densely
(a sliding block per pixel). Used by gabor.py, datasets.py, image_metrics.py.
Angles: theta in [0, pi), ridge direction, measured from the +x (column) axis.
"""
import numpy as np
from scipy import ndimage

import config


def orientation_field(image: np.ndarray, block: int = config.ORIENT_BLOCK
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Dense ridge orientation and coherence.

    Input: uint8 (H, W) print. Output: (theta float32 (H, W) in [0, pi),
    coherence float32 (H, W) in [0, 1], 1 = perfectly parallel ridges).
    Steps (Hong et al. 1998): Gaussian-derivative gradients; per block sums
    Vx = sum 2 Gx Gy, Vy = sum (Gx^2 - Gy^2); gradient angle = atan2(Vx, Vy) / 2;
    ridge angle is perpendicular; smoothed as doubled-angle vectors.
    """
    img = image.astype(np.float32) / 255.0
    gx = ndimage.gaussian_filter(img, config.ORIENT_GRAD_SIGMA, order=(0, 1))
    gy = ndimage.gaussian_filter(img, config.ORIENT_GRAD_SIGMA, order=(1, 0))
    gxx = ndimage.uniform_filter(gx * gx, block)
    gyy = ndimage.uniform_filter(gy * gy, block)
    gxy = ndimage.uniform_filter(gx * gy, block)

    vx, vy = 2 * gxy, gxx - gyy
    coherence = np.hypot(vx, vy) / (gxx + gyy + 1e-8)
    gradient_angle = 0.5 * np.arctan2(vx, vy)
    theta = gradient_angle + np.pi / 2                    # ridges run across the gradient

    # Smooth in the doubled-angle representation, where theta and theta + pi agree.
    sigma = config.ORIENT_SMOOTH_SIGMA * block
    cos2 = ndimage.gaussian_filter(np.cos(2 * theta), sigma)
    sin2 = ndimage.gaussian_filter(np.sin(2 * theta), sigma)
    theta = np.mod(0.5 * np.arctan2(sin2, cos2), np.pi)
    return theta.astype(np.float32), np.clip(coherence, 0, 1).astype(np.float32)


def doubled_angle(theta: np.ndarray) -> np.ndarray:
    """(cos 2theta, sin 2theta) stacked as float32 (2, H, W) in [-1, 1]."""
    return np.stack([np.cos(2 * theta), np.sin(2 * theta)]).astype(np.float32)


def fill_orientation(theta: np.ndarray, valid_mask: np.ndarray,
                     sigma_blocks: float = config.ORIENT_FILL_SIGMA,
                     block: int = config.ORIENT_BLOCK) -> np.ndarray:
    """Fill orientation into pixels where valid_mask is False.

    Normalised convolution of the doubled-angle vectors: Gaussian-smooth
    (vector * valid) and divide by smoothed valid. The kernel doubles until
    every pixel has support. Valid pixels keep their original angle.
    Inputs: theta float (H, W) in [0, pi), valid_mask bool (H, W).
    Output: float32 (H, W) in [0, pi).
    """
    if valid_mask.all() or not valid_mask.any():
        return theta.astype(np.float32)
    weight = valid_mask.astype(np.float32)
    vectors = doubled_angle(theta) * weight
    sigma = sigma_blocks * block
    for _ in range(6):
        support = ndimage.gaussian_filter(weight, sigma)
        if support[~valid_mask].min() > 1e-4:
            break
        sigma *= 2
    cos2 = ndimage.gaussian_filter(vectors[0], sigma) / np.maximum(support, 1e-8)
    sin2 = ndimage.gaussian_filter(vectors[1], sigma) / np.maximum(support, 1e-8)
    filled = np.mod(0.5 * np.arctan2(sin2, cos2), np.pi)
    return np.where(valid_mask, theta, filled).astype(np.float32)


def angle_difference(theta_a: np.ndarray, theta_b: np.ndarray) -> np.ndarray:
    """Absolute ridge-angle difference in radians, in [0, pi/2] (180 deg periodic)."""
    d = np.mod(np.abs(theta_a - theta_b), np.pi)
    return np.minimum(d, np.pi - d)


def orientation_error_deg(theta_a: np.ndarray, theta_b: np.ndarray, region: np.ndarray) -> float:
    """Mean angular difference in degrees over `region` (bool (H, W)); NaN if empty."""
    if not region.any():
        return float("nan")
    return float(np.degrees(angle_difference(theta_a, theta_b)[region].mean()))
