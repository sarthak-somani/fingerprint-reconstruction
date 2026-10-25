"""E17_fingergan_norm as a Reconstructor: E03's FingerGAN (released weights, zero-shot) in the normalised domain.

E03 fed FingerGAN the texture image (sigma 6) of the raw print and handed its output, a ridge map (thin
bright lines on the valleys, black elsewhere), to the evaluator, which MSU-normalises every image again.
E17 changes only the two ends:
- input  'norm': fpr.data.normalization.normalize(degraded) (sigma 2.5, what the evaluator compares), padded with
  grey 128 to at least one window; 'texture6' is E03's input, kept to separate the two changes;
- output: the ridge map (x 255), optionally Gaussian-blurred by `render_sigma` px (thin lines -> ridge-width
  valleys) and then normalize()d, so it is a normalised print (output_normalized, measured as stored);
- `paste_back`: true keeps the normalised input outside the mask (the repair protocol), false = whole print
  (the baselines' agreed protocol).
The variant is chosen on v3 val by sweep.py (selected.json). The network and its sliding windows are E03's.
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import config
from experiments.baselines.E03_fingergan.reconstructor import FingerGanReconstructor
from experiments.baselines.E04_msu_stft import preprocess
from fpr.data.normalization import normalize
from fpr.models.base import paste_back, Reconstructor

HERE = Path(__file__).resolve().parent
E03 = HERE.parent / "E03_fingergan"


def render(ridge_map: np.ndarray, sigma: float, normalized: bool) -> np.ndarray:
    """float (H, W) FingerGAN output in [0, 1] -> uint8 (H, W) print (bright valleys, dark ridges)."""
    image = ridge_map * 255
    if sigma > 0:
        image = ndimage.gaussian_filter(image, sigma)
        image = (image - image.min()) / (np.ptp(image) + 1e-6) * 255   # blurring lowers the contrast
    image = np.clip(np.round(image), 0, 255).astype(np.uint8)
    return normalize(image) if normalized else image


class FingerGanNormReconstructor(Reconstructor):
    name = "fingergan_norm"

    def __init__(self, params: dict, weights: Path | bool | None = None, selected: dict | None = None):
        self.params = params
        self.gan = FingerGanReconstructor(params, weights=E03 / params["weights"] if weights is None else weights)
        self.device = self.gan.device
        if selected is None:
            path = HERE / params["inference"]["selected"]
            selected = json.loads(path.read_text()) if path.exists() else params["inference"]["default"]
        self.selected = selected

    def num_parameters(self) -> int:
        return self.gan.num_parameters()

    def ridge_map(self, degraded: np.ndarray, input_kind: str) -> np.ndarray:
        """uint8 (H, W) damaged print -> float (H, W) in [0, 1]: FingerGAN's ridge map of the chosen input."""
        h, w = degraded.shape
        size = self.params["window"]
        if input_kind == "norm":
            padded = np.full((max(h, size), max(w, size)), config.NORMALIZE_BACKGROUND, np.uint8)
            padded[:h, :w] = normalize(degraded)                    # normalise first, pad with its grey
            image = padded / 255.0
        else:
            assert input_kind == "texture6", input_kind
            padded = np.full((max(h, size), max(w, size)), config.BACKGROUND_VALUE, np.uint8)
            padded[:h, :w] = degraded                               # E03's preprocessing, unchanged
            image = np.round(preprocess.fast_cartoon_texture(padded, sigma=6)) / 255.0
        return self.gan.enhance(image)[:h, :w]

    def finish(self, ridge_map: np.ndarray, degraded: np.ndarray, mask: np.ndarray, render_sigma: float,
               pasted: bool) -> np.ndarray:
        """Ridge map -> the stored output: a normalised print, pasted back into normalize(degraded) or not."""
        out = render(ridge_map, render_sigma, normalized=True)
        return paste_back(normalize(degraded), out, mask) if pasted else out

    def repair(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        s = self.selected
        return self.finish(self.ridge_map(degraded, s["input"]), degraded, mask, s["render_sigma"], s["paste_back"])

    def reconstruct(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        s = self.selected
        return render(self.ridge_map(degraded, s["input"]), s["render_sigma"], normalized=True)


def build_reconstructor(params: dict) -> Reconstructor:
    return FingerGanNormReconstructor(params)
