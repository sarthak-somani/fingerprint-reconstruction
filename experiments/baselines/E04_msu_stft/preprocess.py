"""MSU-LatentAFIS preprocessing, ported to Python 3: cartoon-texture decomposition and STFT enhancement.

Source: prip-lab/MSU-LatentAFIS, extraction/preprocessing.py (MIT licence, (c) 2019 ctymoszek):
FastCartoonTexture, STFT and their helpers, with the same logic and constants. Used by
E04_msu_stft and E05_msu_aec (texture sigma 2.5, then STFT) and E03_fingergan (texture sigma 6, as FingerGAN's README asks).
"""
import math

import cv2
import numpy as np

import config


def pad_to_multiple(image: np.ndarray, block: int) -> np.ndarray:
    """uint8 (H, W) -> (H', W') with H', W' multiples of `block`, padded with white at the bottom/right.

    Stands in for MSU's adjust_image_size, which under Python 2 *cropped* to a multiple of the
    block size; padding keeps every probe pixel, and callers crop back to (H, W).
    """
    h, w = image.shape
    return cv2.copyMakeBorder(image, 0, -h % block, 0, -w % block, cv2.BORDER_CONSTANT,
                              value=config.BACKGROUND_VALUE)


def lowpass_filtering(image: np.ndarray, lowpass: np.ndarray) -> np.ndarray:
    """MSU LowpassFiltering: multiply the centred spectrum by `lowpass` (FFT size >= image)."""
    h, w = image.shape
    h2, w2 = lowpass.shape
    image = cv2.copyMakeBorder(image, 0, h2 - h, 0, w2 - w, cv2.BORDER_CONSTANT, value=0)
    spectrum = np.fft.fftshift(np.fft.fft2(image)) * lowpass
    return np.real(np.fft.ifft2(np.fft.fftshift(spectrum)))[:h, :w]


def gradient_norm(image: np.ndarray) -> np.ndarray:
    """MSU compute_gradient_norm: |grad| + 1e-6, float32 (H, W)."""
    gx, gy = np.gradient(image.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy) + 0.000001


def fast_cartoon_texture(image: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    """MSU FastCartoonTexture (Buades et al. 2010): the texture part of the print.

    Input: (H, W) grey print. Output: float (H, W) in [0, 255], texture centred on 127.5
    (the ridge pattern without the slowly varying background and stains).
    """
    image = image.astype(np.float32)
    h, w = image.shape
    fft_size = max(2 ** int(math.ceil(math.log(h, 2))), 2 ** int(math.ceil(math.log(w, 2))))
    x, y = np.meshgrid(range(-fft_size // 2, fft_size // 2), range(-fft_size // 2, fft_size // 2))
    r = (np.sqrt(x * x + y * y) + 0.0001) / fft_size
    lowpass = 1. / (1 + (2 * math.pi * r * sigma) ** 4)
    image_low = lowpass_filtering(image, lowpass)

    grad_image = lowpass_filtering(gradient_norm(image), lowpass)
    grad_low = lowpass_filtering(gradient_norm(image_low), lowpass)
    diff = grad_image - grad_low
    strong = np.abs(grad_image) > 1
    diff[strong] = diff[strong] / np.abs(grad_image)[strong]
    diff[~strong] = 0

    cmin, cmax = 0.3, 0.7                     # relative loss of gradient: where the cartoon takes over
    weight = np.clip((diff - cmin) / (cmax - cmin), 0, 1)
    cartoon = weight * image_low + (1 - weight) * image
    lim = 20
    return np.clip((image - cartoon + lim) * 255 / (2 * lim), 0, 255)


def stft_enhance(image: np.ndarray) -> np.ndarray:
    """MSU STFT: block-wise band-pass filtering (64 px windows, 16 px blocks, ridge spacing 3-18 px).

    Input: float (H, W) image. Output: float (H, W) in [0, 255], normalised to median 127, std 14.
    """
    patch_size, block_size = 64, 16
    ovp_size = (patch_size - block_size) // 2
    h0, w0 = image.shape
    image = cv2.copyMakeBorder(image, ovp_size, ovp_size, ovp_size, ovp_size, cv2.BORDER_CONSTANT, value=0)
    h, w = image.shape
    blk_h, blk_w = (h - patch_size) // block_size, (w - patch_size) // block_size

    rmin, rmax = 3, 18                        # allowed ridge spacing in px
    flow, fhigh = patch_size / rmax, patch_size / rmin
    half = patch_size // 2
    x, y = np.meshgrid(range(-half, half), range(-half, half))
    r = np.sqrt(x * x + y * y) + 0.0001
    band_pass = 1. / (1 + (r / fhigh) ** 4) * 1. / (1 + (flow / r) ** 4)   # Butterworth low * high
    weight = np.exp(-(x * x + y * y) / (patch_size / 3) ** 2)

    rec = np.zeros((h, w))
    for i in range(blk_h):
        for j in range(blk_w):
            window = (slice(i * block_size, i * block_size + patch_size),
                      slice(j * block_size, j * block_size + patch_size))
            patch = image[window] - np.median(image[window])
            filtered = band_pass * np.fft.fftshift(np.fft.fft2(patch))
            filtered = filtered / (np.linalg.norm(filtered) + 0.0001)
            rec[window] += np.real(np.fft.ifft2(np.fft.ifftshift(filtered))) * weight

    rec = rec[ovp_size:ovp_size + h0, ovp_size:ovp_size + w0]
    rec = (rec - np.median(rec)) / (np.std(rec) + 0.000001)
    return np.clip(rec * 14 + 127, 0, 255)
