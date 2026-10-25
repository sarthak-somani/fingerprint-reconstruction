"""MSU normalisation: the texture part of MSU-LatentAFIS's cartoon-texture decomposition.

normalize(image) is what evaluation/evaluate.py applies to every image before measuring
and what E12 applies to every image it sees in training and repair: the ridge pattern
without background, brightness and stain shading, 8-bit, flat regions at config.NORMALIZE_BACKGROUND.
It is deterministic, so a normalised copy on disk (preprocessing/10_normalize.py) equals
normalising on the fly.
"""
from pathlib import Path

import numpy as np

import config
from fpr.data import image_io


def normalize(image: np.ndarray, sigma: float = config.NORMALIZE_SIGMA) -> np.ndarray:
    """uint8 (H, W) grey print -> uint8 (H, W) texture component (Buades et al.; MSU's port, sigma 2.5)."""
    from experiments.baselines.E04_msu_stft.preprocess import fast_cartoon_texture   # imported here: cv2 / scipy
    return np.round(fast_cartoon_texture(image, sigma)).astype(np.uint8)


def normalize_file(src: Path, dst: Path) -> None:
    """normalize() of one PNG file into another (skipped if dst exists). For process pools."""
    if not Path(dst).exists():
        image_io.save_png(dst, normalize(image_io.load_gray(src)))
