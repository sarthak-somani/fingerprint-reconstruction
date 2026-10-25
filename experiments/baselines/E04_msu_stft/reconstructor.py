"""E04_msu_stft as a Reconstructor: MSU-LatentAFIS's non-learned enhancement, on the whole print.

Cartoon-texture decomposition (sigma 2.5) -> STFT band-pass enhancement, the input stage of MSU's
latent pipeline (extraction_latent.py L97-98). Its learned autoencoder is E05_msu_aec: the released
weights are no longer downloadable. Like the method itself, it ignores the mask: repair() does NOT
paste back (see README.md).
"""
import numpy as np

from experiments.baselines.E04_msu_stft import preprocess
from fpr.models.base import Reconstructor


def msu_stft(image: np.ndarray, sigma: float, block: int) -> np.ndarray:
    """uint8 (H, W) print -> float (H, W) in [0, 255]: MSU's STFT-enhanced texture image."""
    h, w = image.shape
    texture = preprocess.fast_cartoon_texture(preprocess.pad_to_multiple(image, block), sigma=sigma)
    return preprocess.stft_enhance(texture)[:h, :w]


class MsuStftReconstructor(Reconstructor):
    name = "msu_stft"

    def __init__(self, params: dict):
        self.params, self.device = params, "cpu"

    def reconstruct(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """uint8 (H, W) damaged print -> uint8 (H, W) enhanced print. `mask` is not used."""
        enhanced = msu_stft(degraded, self.params["texture_sigma"], self.params["block"])
        return np.clip(np.round(enhanced), 0, 255).astype(np.uint8)

    def repair(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Whole-image enhancement, no paste-back: this is how the MSU system uses it."""
        return self.reconstruct(degraded, mask)


def build_reconstructor(params: dict) -> Reconstructor:
    return MsuStftReconstructor(params)
