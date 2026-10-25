"""E10_ctmsr as a Reconstructor: one-step CTMSR repair (consistency.one_step) of every damaged tile.

E02's tiling, batching, fixed seed and paste-back (fpr/models/reconstructors.py), with one network
call per batch instead of the reverse chain. Also used by E11_ctmsr_dtm.
"""
from pathlib import Path

import torch

from experiments.one_step.E10_ctmsr import consistency
from fpr.models.reconstructors import ResShiftReconstructor

HERE = Path(__file__).resolve().parent


class CtmsrReconstructor(ResShiftReconstructor):
    name = "ctmsr"

    def __init__(self, params: dict, checkpoint: Path | None):
        super().__init__(params, params["inference"], checkpoint)
        self.sigma_max = params["consistency"]["sigma_max"]

    def sample_batch(self, y0: torch.Tensor, mask: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x0 = consistency.one_step(self.net, y0, cond, self.sigma_max)
        return mask * x0 + (1 - mask) * y0


def build_reconstructor(params: dict, folder: Path = HERE, random_init: bool = False) -> CtmsrReconstructor:
    return CtmsrReconstructor(params, None if random_init else folder / params["inference"]["checkpoint"])
