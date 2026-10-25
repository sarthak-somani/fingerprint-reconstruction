"""E18_msu_stft_norm as a Reconstructor: E04's STFT enhancement on the MSU-normalised print, stored as is.

E04 = STFT(texture_2.5(raw print)), and the evaluator then normalises that output again (texture 2.5 of a
band-passed image). E18 feeds STFT the normalised print, fpr.data.normalization.normalize(degraded) (padded with
its grey 128), and its output is measured as stored (params.json output_normalized, evaluate.py
--rec-normalized): no second filtering. The input change alone is numerically negligible (E04's first stage
already is the texture step; outputs correlate 0.999 after the evaluator's normalisation, see README), so the
experiment is the as-stored measurement. Whole print, no paste-back, as E04.
"""
import numpy as np

import config
from experiments.baselines.E04_msu_stft import preprocess
from fpr.data.normalization import normalize
from fpr.models.base import Reconstructor


def stft_of_normalized(normalized: np.ndarray, block: int) -> np.ndarray:
    """uint8 (H, W) normalised print -> uint8 (H, W) STFT-enhanced print (median 127, std about 14)."""
    h, w = normalized.shape
    padded = np.pad(normalized, ((0, -h % block), (0, -w % block)), constant_values=config.NORMALIZE_BACKGROUND)
    out = preprocess.stft_enhance(padded.astype(np.float32))[:h, :w]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


class MsuStftNormReconstructor(Reconstructor):
    name = "msu_stft_norm"

    def __init__(self, params: dict):
        self.params, self.device = params, "cpu"

    def reconstruct(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return stft_of_normalized(normalize(degraded), self.params["block"])

    def repair(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Whole-print enhancement, no paste-back (as E04)."""
        return self.reconstruct(degraded, mask)


def build_reconstructor(params: dict) -> Reconstructor:
    return MsuStftNormReconstructor(params)
