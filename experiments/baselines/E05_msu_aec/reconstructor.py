"""E05_msu_aec as a Reconstructor: MSU's enhancement autoencoder (retrained) on the whole print.

As MSU's pipeline (extraction_latent.py L97-105, enhancement_AEC.run_whole_image): cartoon-texture
(sigma 2.5) -> STFT -> /128 - 1 -> the autoencoder on the whole image -> min-max to 0-255.
Like the method itself, it ignores the mask: repair() does NOT paste back (see README.md).
"""
from pathlib import Path

import numpy as np
import torch

from experiments.baselines.E04_msu_stft.reconstructor import msu_stft
from experiments.baselines.E05_msu_aec.model import EnhancementAEC
from fpr.models.base import Reconstructor

HERE = Path(__file__).resolve().parent
STRIDE = 32                                     # five stride-2 layers: whole-image input must divide by 32


class MsuAecReconstructor(Reconstructor):
    name = "msu_aec"

    def __init__(self, params: dict, checkpoint: Path | bool | None = None):
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = EnhancementAEC(params["model"]["nf"]).to(self.device).eval()
        if checkpoint is not False:                                # False: untrained net (tests)
            state = torch.load(checkpoint or HERE / params["inference"]["checkpoint"],
                               map_location=self.device, weights_only=True)
            self.net.load_state_dict(state["model"])

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    @torch.no_grad()
    def reconstruct(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """uint8 (H, W) damaged print -> uint8 (H, W) enhanced print. `mask` is not used."""
        h, w = degraded.shape
        stft = msu_stft(degraded, self.params["texture_sigma"], self.params["block"])
        padded = np.full((-(-h // STRIDE) * STRIDE, -(-w // STRIDE) * STRIDE), 127.0, np.float32)
        padded[:h, :w] = stft                                      # 127 = flat STFT background
        x = torch.from_numpy(padded / 128.0 - 1)[None, None].to(self.device)
        out = self.net(x)[0, 0, :h, :w].cpu().numpy()
        out = (out - out.min()) / (out.max() - out.min() + 1e-8) * 255
        return np.clip(np.round(out), 0, 255).astype(np.uint8)

    def repair(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Whole-image enhancement, no paste-back: this is how the MSU system uses it."""
        return self.reconstruct(degraded, mask)


def build_reconstructor(params: dict) -> Reconstructor:
    return MsuAecReconstructor(params)
