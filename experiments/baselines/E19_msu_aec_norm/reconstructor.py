"""E19_msu_aec_norm as a Reconstructor: E05's MSU autoencoder, trained on our pairs, on normalised prints.

repair(degraded, mask): MSU-normalise the damaged print (fpr.data.normalization, the evaluator's own
normalisation) -> [-1, 1] -> the autoencoder on the whole print (padded to a multiple of 32 with grey,
which is 0 in the model's range) -> back to 0-255. The output is a normalised print (params.json
output_normalized), measured as stored. `paste_back` (chosen on v3 val by sweep.py, written to
selected.json) keeps the normalised input outside the mask; false = whole print, the baselines' protocol.
"""
import json
from pathlib import Path

import numpy as np
import torch

import config
from experiments.baselines.E05_msu_aec.model import EnhancementAEC
from fpr.data.normalization import normalize
from fpr.models.base import paste_back, Reconstructor

HERE = Path(__file__).resolve().parent
STRIDE = 32                                     # five stride-2 layers: whole-image input must divide by 32


class MsuAecNormReconstructor(Reconstructor):
    name = "msu_aec_norm"

    def __init__(self, params: dict, checkpoint: Path | bool | None = None, selected: dict | None = None):
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = EnhancementAEC(params["model"]["nf"]).to(self.device).eval()
        if checkpoint is not False:                                # False: untrained net (tests)
            state = torch.load(checkpoint or HERE / params["inference"]["checkpoint"],
                               map_location=self.device, weights_only=False)
            self.net.load_state_dict(state["model"])
        if selected is None:
            path = HERE / params["inference"]["selected"]
            selected = json.loads(path.read_text()) if path.exists() else {"paste_back": False}
        self.paste_back = bool(selected["paste_back"])

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    @torch.no_grad()
    def enhance_normalized(self, normalized: np.ndarray) -> np.ndarray:
        """uint8 (H, W) normalised print -> uint8 (H, W) enhanced normalised print (whole image)."""
        h, w = normalized.shape
        padded = np.full((-(-h // STRIDE) * STRIDE, -(-w // STRIDE) * STRIDE), config.NORMALIZE_BACKGROUND,
                         np.float32)
        padded[:h, :w] = normalized
        x = torch.from_numpy(padded / 127.5 - 1)[None, None].to(self.device)
        out = (self.net(x)[0, 0, :h, :w].cpu().numpy() + 1) * 127.5
        return np.clip(np.round(out), 0, 255).astype(np.uint8)

    def reconstruct(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.enhance_normalized(normalize(degraded))

    def repair(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Raw damaged print -> normalised repaired print (whole print, or pasted back if selected)."""
        normalized = normalize(degraded)
        enhanced = self.enhance_normalized(normalized)
        return paste_back(normalized, enhanced, mask) if self.paste_back else enhanced


def build_reconstructor(params: dict) -> Reconstructor:
    return MsuAecNormReconstructor(params)
