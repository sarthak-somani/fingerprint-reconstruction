"""Classical baseline: orientation- and frequency-tuned Gabor enhancement.

Hong, Wan and Jain, "Fingerprint Image Enhancement: Algorithm and Performance
Evaluation", IEEE TPAMI 1998: normalise -> orientation -> ridge frequency ->
Gabor filter bank. Orientation and frequency are filled into the damaged
region from the intact surroundings, then the result is pasted back.
"""
import cv2
import numpy as np
from scipy import ndimage, signal

import config
from fpr.data import foreground, orientation
from fpr.models.base import Reconstructor


def normalise(image: np.ndarray) -> np.ndarray:
    """Hong et al. eq. 1: shift each pixel so the image has mean M0 and variance V0. float32."""
    img = image.astype(np.float32)
    mean, var = img.mean(), img.var() + 1e-8
    spread = np.sqrt(config.GABOR_NORM_VAR * (img - mean) ** 2 / var)
    return np.where(img > mean, config.GABOR_NORM_MEAN + spread, config.GABOR_NORM_MEAN - spread)


def block_frequency(image: np.ndarray, theta: float, centre: tuple[int, int]) -> float:
    """Ridge frequency (1/px) at one block from its x-signature; NaN if not measurable.

    The window is sampled across the ridges (FREQ_WINDOW_PX[0] long) and
    averaged along them; the mean spacing of the signature's minima (ridges
    are dark) is the ridge period.
    """
    across_len, along_len = config.FREQ_WINDOW_PX
    normal = theta + np.pi / 2
    u = np.arange(across_len) - across_len / 2                        # across ridges
    v = np.arange(along_len) - along_len / 2                          # along ridges
    uu, vv = np.meshgrid(u, v, indexing="ij")
    cols = centre[1] + uu * np.cos(normal) + vv * np.cos(theta)
    rows = centre[0] + uu * np.sin(normal) + vv * np.sin(theta)
    samples = ndimage.map_coordinates(image, [rows, cols], order=1, mode="nearest")
    x_signature = samples.mean(axis=1)

    low, high = config.FREQ_PERIOD_RANGE_PX
    minima, _ = signal.find_peaks(-x_signature, distance=low)
    if len(minima) < 2:
        return float("nan")
    period = (minima[-1] - minima[0]) / (len(minima) - 1)
    return 1.0 / period if low <= period <= high else float("nan")


def frequency_field(image: np.ndarray, theta: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Dense ridge frequency (1/px), float32 (H, W).

    Measured at block centres inside `valid`; missing blocks get the median
    of the measured ones; then smoothed and upsampled to pixels.
    """
    block = config.ORIENT_BLOCK
    rows = np.arange(block // 2, image.shape[0], block)
    cols = np.arange(block // 2, image.shape[1], block)
    freq = np.full((len(rows), len(cols)), np.nan, np.float32)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if valid[r, c]:
                freq[i, j] = block_frequency(image, theta[r, c], (r, c))
    measured = np.isfinite(freq)
    fallback = np.median(freq[measured]) if measured.any() else 1.0 / 9   # ~9 px period at 500 ppi
    freq[~measured] = fallback
    freq = ndimage.gaussian_filter(freq, 1.0)
    height, width = image.shape
    return cv2.resize(freq, (width, height), interpolation=cv2.INTER_LINEAR)


def gabor_kernel(theta: float, frequency: float) -> np.ndarray:
    """Even-symmetric Gabor filter (Hong et al. eq. 8) tuned to ridge angle theta.

    The cosine runs across the ridges (along the normal), the Gaussian
    envelope has sigma GABOR_SIGMA_PX. Zero-mean so flat areas give 0.
    """
    sigma = config.GABOR_SIGMA_PX
    half = int(np.ceil(3 * sigma))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    normal = theta + np.pi / 2
    across = xx * np.cos(normal) + yy * np.sin(normal)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * frequency * across)
    return (kernel - kernel.mean()).astype(np.float32)


def gabor_enhance(image: np.ndarray, theta: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Filter with a bank (N_ORIENT x N_FREQ) and keep, per pixel, its own filter's response.

    Output: uint8 (H, W), dark ridges on white like the input.
    """
    img = normalise(image) - config.GABOR_NORM_MEAN
    angle_bins = np.round(theta / np.pi * config.GABOR_N_ORIENT).astype(int) % config.GABOR_N_ORIENT
    freq_levels = np.quantile(freq, np.linspace(0, 1, config.GABOR_N_FREQ + 2)[1:-1])
    freq_bins = np.abs(freq[..., None] - freq_levels).argmin(axis=-1)

    response = np.zeros(image.shape, np.float32)
    for a in range(config.GABOR_N_ORIENT):
        for f, level in enumerate(freq_levels):
            pixels = (angle_bins == a) & (freq_bins == f)
            if pixels.any():
                kernel = gabor_kernel(a * np.pi / config.GABOR_N_ORIENT, level)
                response[pixels] = cv2.filter2D(img, -1, kernel)[pixels]
    # Contrast relative to the local response strength, so weak (damaged) areas stay visible.
    local_rms = np.sqrt(ndimage.gaussian_filter(response ** 2, config.GABOR_LOCAL_SIGMA_PX)) + 1e-6
    out = 127.5 + 127.5 * np.tanh(config.GABOR_CONTRAST * response / local_rms)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


class GaborReconstructor(Reconstructor):
    name = "gabor"

    def reconstruct(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Enhance the whole print with orientation/frequency filled in from outside the mask."""
        fg = foreground.foreground_mask(degraded)
        valid = fg & ~mask
        theta, _ = orientation.orientation_field(degraded)
        theta = orientation.fill_orientation(theta, valid)
        freq = frequency_field(degraded, theta, valid)
        return gabor_enhance(degraded, theta, freq)
