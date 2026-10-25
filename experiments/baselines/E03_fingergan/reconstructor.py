"""E03_fingergan as a Reconstructor: FingerGAN's released weights, zero-shot, on the whole print.

Follows HubYZ/LatentEnhancement experiment/latent_fingerprint_enhancement.py: texture image
(cartoon-texture, sigma 6) read as /255 -> 192 px windows every `step` px -> overlapping outputs
averaged -> x255. Like the method itself, it enhances the whole print and ignores the mask:
repair() does NOT paste back (see README.md).
"""
from pathlib import Path

import numpy as np
import torch

import config
from experiments.baselines.E03_fingergan.network import FingerGAN
from experiments.baselines.E04_msu_stft import preprocess
from fpr.models.base import Reconstructor

HERE = Path(__file__).resolve().parent


def window_ends(length: int, window: int, step: int) -> list[int]:
    """End offsets of the windows along one axis, as in FingerGAN's LatentDataset (both directions)."""
    forward = list(range(window, length + 1, step))
    return forward + list(range(length, forward[-1] - 1, -step))


class FingerGanReconstructor(Reconstructor):
    name = "fingergan"

    def __init__(self, params: dict, weights: Path | bool | None = None):
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = FingerGAN(in_channels=1).to(self.device).eval()
        if weights is not False:                                   # False: untrained net (tests)
            state = torch.load(weights or HERE / params["weights"], map_location=self.device, weights_only=False)
            self.net.load_state_dict(state["UNet_model_state"])

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    @torch.no_grad()
    def enhance(self, texture: np.ndarray) -> np.ndarray:
        """float (H, W) in [0, 1], H, W >= window -> float (H, W) in [0, 1] (averaged windows)."""
        size, step = self.params["window"], self.params["step"]
        image = torch.from_numpy(texture.astype(np.float32)).to(self.device)
        ends = [(r, c) for r in window_ends(image.shape[0], size, step)
                for c in window_ends(image.shape[1], size, step)]
        total, count = torch.zeros_like(image), torch.zeros_like(image)
        for start in range(0, len(ends), self.params["batch"]):
            chunk = ends[start:start + self.params["batch"]]
            patches = torch.stack([image[r - size:r, c - size:c] for r, c in chunk])[:, None]
            out = self.net(patches)[:, 0]
            for (r, c), patch in zip(chunk, out):
                total[r - size:r, c - size:c] += patch
                count[r - size:r, c - size:c] += 1
        return (total / count.clamp(min=1)).cpu().numpy()

    def reconstruct(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """uint8 (H, W) damaged print -> uint8 (H, W) enhanced print. `mask` is not used."""
        h, w = degraded.shape
        size = self.params["window"]
        padded = np.full((max(h, size), max(w, size)), config.BACKGROUND_VALUE, np.uint8)
        padded[:h, :w] = degraded                                  # FingerGAN needs >= 192 px per side
        texture = preprocess.fast_cartoon_texture(padded, sigma=self.params["texture_sigma"])
        texture = np.round(texture) / 255.0                        # as the texture PNG read by their script
        enhanced = self.enhance(texture)[:h, :w] * 255
        if self.params["invert"]:
            enhanced = 255 - enhanced
        return np.clip(np.round(enhanced), 0, 255).astype(np.uint8)

    def repair(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Whole-image enhancement, no paste-back: this is how FingerGAN is used."""
        return self.reconstruct(degraded, mask)


def build_reconstructor(params: dict) -> Reconstructor:
    return FingerGanReconstructor(params)
