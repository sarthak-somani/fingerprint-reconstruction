"""The diffusion models as Reconstructors: repair full prints tile by tile with a trained ResShift network.

ResShiftReconstructor   ResShift's reverse chain (Algorithm 2), 15 steps          E02, E06-E08
CcdfReconstructor       CCDF warm start at t0 < T (fpr/models/ccdf.py)            E09, E21-E23, E26, E27
NormalizedCcdfReconstructor  CCDF on MSU-normalised prints, normalised output    E12-E16, E24, E25

Every 128 px tile that contains damage is sampled, all tiles of a print in GPU batches; tiles are
blended and Reconstructor.repair pastes back, so pixels outside the mask never change. Each
experiment's reconstructor.py builds one of these from its params.json.
"""
from pathlib import Path

import numpy as np
import torch

import config as repo_config
from fpr.data import foreground
from fpr.data.normalization import normalize
from fpr.models import ccdf, inputs, resshift
from fpr.models.base import paste_back, reconstruct_full_image_batched, Reconstructor
from fpr.models.schedule import build_schedule, DIFFUSION
from fpr.models.unet import UNet


def numbers(block: dict) -> dict:
    """A params.json block without its '_' comment keys."""
    return {k: v for k, v in block.items() if not k.startswith("_")}


def resolve(path: str, exp_dir: Path) -> Path:
    """A params.json path: relative to the repo root if it starts with 'experiments/', else to exp_dir."""
    return repo_config.REPO_ROOT / path if Path(path).parts[0] == "experiments" else exp_dir / path


class ResShiftReconstructor(Reconstructor):
    name = "resshift"

    def __init__(self, model_params: dict, inference: dict, checkpoint: Path | None):
        """model_params: a params.json with 'model' and 'diffusion'; inference: tile_batch, seed (and
        sampling_steps for the plain chain); checkpoint: EMA weights saved by fpr/training (None = random
        initialisation, for smoke tests only)."""
        # The samplers use E02's schedule (fpr/models/schedule.py), so the model must use the same one
        # (E10/E11's one-step models have no diffusion block).
        assert numbers(model_params.get("diffusion", DIFFUSION)) == numbers(DIFFUSION), "diffusion differs from E02's"
        self.params = inference
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = UNet(**numbers(model_params["model"])).to(self.device).eval()
        self.trained_steps = 0
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location=self.device, weights_only=True)
            self.net.load_state_dict(state["ema"])              # EMA weights, as saved by the trainers
            self.trained_steps = state["step"]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def sample_batch(self, y0: torch.Tensor, mask: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """(B, 1, P, P) repaired tiles in [-1, 1]: ResShift's full reverse chain."""
        return resshift.sample(self.net, y0, mask, cond, steps=self.params["sampling_steps"])

    @torch.no_grad()
    def sample_tiles(self, tiles: np.ndarray, tile_masks: np.ndarray, tile_fgs: np.ndarray) -> np.ndarray:
        """uint8 (N, P, P) damaged tiles -> uint8 (N, P, P) repaired tiles, in batches."""
        out, batch = [], self.params["tile_batch"]
        for start in range(0, len(tiles), batch):
            parts = [inputs.tile_inputs(t, m, f) for t, m, f in
                     zip(*(a[start:start + batch] for a in (tiles, tile_masks, tile_fgs)))]
            y0, mask, cond = (torch.cat(p).to(self.device) for p in zip(*parts))
            x0 = self.sample_batch(y0, mask, cond)
            out.append((inputs.from_model_range(x0) * 255).round().byte().cpu().numpy()[:, 0])
        return np.concatenate(out)

    def reconstruct(self, degraded: np.ndarray, mask: np.ndarray,
                    background: int = repo_config.BACKGROUND_VALUE) -> np.ndarray:
        """uint8 (H, W) damaged print + bool (H, W) mask -> uint8 (H, W) repaired print (before paste-back)."""
        torch.manual_seed(self.params["seed"])               # the same repair every run
        size = max(degraded.shape[0], 128), max(degraded.shape[1], 128)
        fg = np.zeros(size, bool)
        fg[:degraded.shape[0], :degraded.shape[1]] = foreground.foreground_mask(degraded)

        def batch_fn(tiles, tile_masks, corners):
            tile_fgs = np.stack([fg[t:t + tiles.shape[1], l:l + tiles.shape[2]] for t, l in corners])
            return self.sample_tiles(tiles, tile_masks, tile_fgs)

        return reconstruct_full_image_batched(batch_fn, degraded, mask, background=background)


class CcdfReconstructor(ResShiftReconstructor):
    """A trained ResShift network sampled CCDF-style: starting estimate ('e02_x0' / 'one_call', 'y0' or
    'regressor') -> one-jump forward diffusion to t0 -> reverse chain t0..1, optional projection."""
    name = "ccdf"

    def __init__(self, model_params: dict, inference: dict, checkpoint: Path | None, config: dict,
                 regressor: torch.nn.Module | None = None):
        super().__init__(model_params, inference, checkpoint)
        self.config = config                                   # {'init', 't0', 'project'} (selected.json)
        self.schedule = build_schedule()
        self.regressor = regressor

    def sample_batch(self, y0: torch.Tensor, mask: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x_init = ccdf.starting_estimate(self.config["init"], self.net, self.regressor, y0, cond, self.schedule)
        return ccdf.sample(self.net, y0, mask, cond, x_init, self.config["t0"], self.config["project"],
                           self.schedule)

    def num_parameters(self) -> int:
        extra = 0 if self.regressor is None else sum(p.numel() for p in self.regressor.parameters())
        return super().num_parameters() + extra


class NormalizedCcdfReconstructor(CcdfReconstructor):
    """CCDF in the MSU-normalised domain: normalise the damaged print, repair, paste back into the
    *normalised* print. Its output is a normalised print (params.json output_normalized), evaluated as stored."""
    name = "norm_ccdf"

    def reconstruct_normalized(self, normalized: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """uint8 (H, W) normalised damaged print -> uint8 (H, W) repaired, before paste-back."""
        return super().reconstruct(normalized, mask, background=repo_config.NORMALIZE_BACKGROUND)

    def repair_normalized(self, normalized: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Repair an already normalised print (the v3 val sweep reads preprocessing script 10's copies)."""
        return paste_back(normalized, self.reconstruct_normalized(normalized, mask), mask)

    def reconstruct(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.reconstruct_normalized(normalize(degraded), mask)

    def repair(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Raw damaged print -> repaired *normalised* print (pixels outside the mask = normalised input)."""
        return self.repair_normalized(normalize(degraded), mask)
