"""E20_fingergan_finetune as a Reconstructor: FingerGAN fine-tuned on our normalised pairs (train.py).

repair(degraded, mask): normalize(degraded), padded with grey 128 to at least one window -> E03's sliding
192 px windows (step 10, overlaps averaged) with the fine-tuned generator -> x 255. It was trained to output
the normalised clean print, so its output is a normalised print (output_normalized), measured as stored,
with no rendering step (unlike E17). `paste_back` comes from the val sweep (selected.json).
"""
import json
from pathlib import Path

import numpy as np

import config
from experiments.baselines.E03_fingergan.reconstructor import FingerGanReconstructor
from fpr.data.normalization import normalize
from fpr.models.base import paste_back, Reconstructor

HERE = Path(__file__).resolve().parent


class FingerGanFinetunedReconstructor(Reconstructor):
    name = "fingergan_finetune"

    def __init__(self, params: dict, checkpoint: Path | bool | None = None, selected: dict | None = None):
        self.params = params
        weights = HERE / params["inference"]["checkpoint"] if checkpoint is None else checkpoint
        self.gan = FingerGanReconstructor(params, weights=weights)
        self.device = self.gan.device
        if selected is None:
            path = HERE / params["inference"]["selected"]
            selected = json.loads(path.read_text()) if path.exists() else {"paste_back": False}
        self.paste_back = bool(selected["paste_back"])

    def num_parameters(self) -> int:
        return self.gan.num_parameters()

    def enhance_normalized(self, normalized: np.ndarray) -> np.ndarray:
        """uint8 (H, W) normalised print -> uint8 (H, W) repaired normalised print (whole image)."""
        h, w = normalized.shape
        size = self.params["window"]
        padded = np.full((max(h, size), max(w, size)), config.NORMALIZE_BACKGROUND, np.uint8)
        padded[:h, :w] = normalized
        out = self.gan.enhance(padded / 255.0)[:h, :w] * 255
        return np.clip(np.round(out), 0, 255).astype(np.uint8)

    def reconstruct(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.enhance_normalized(normalize(degraded))

    def repair(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        normalized = normalize(degraded)
        out = self.enhance_normalized(normalized)
        return paste_back(normalized, out, mask) if self.paste_back else out


def build_reconstructor(params: dict) -> Reconstructor:
    return FingerGanFinetunedReconstructor(params)
