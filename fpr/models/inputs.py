"""Turn dataset items and repair tiles into the tensors E02's model sees.

The model works in [-1, 1] (as ResShift). cond = [damaged y0, mask, cos 2theta, sin 2theta],
4 channels; the orientation field is measured on the damaged image outside the mask and
filled into it (fpr.data.datasets.filled_orientation). Used by the trainers (fpr/training) and fpr/models/reconstructors.py,
so training and repair build the inputs the same way.
"""
import numpy as np
import torch

from fpr.data import datasets


def to_model_range(image01: torch.Tensor) -> torch.Tensor:
    """[0, 1] -> [-1, 1]."""
    return image01 * 2 - 1


def from_model_range(image: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]."""
    return (image.clamp(-1, 1) + 1) / 2


def batch_inputs(batch: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    """(x0, y0, mask, cond) from a FixedTrainDataset / TrainPatchDataset batch.

    x0, y0, mask: (B, 1, P, P), images in [-1, 1], mask in {0, 1}; cond: (B, 4, P, P).
    """
    x0 = to_model_range(batch["clean"].to(device))
    y0 = to_model_range(batch["degraded"].to(device))
    mask = batch["mask"].to(device)
    cond = torch.cat([y0, mask, batch["orientation"].to(device)], dim=1)
    return x0, y0, mask, cond


def tile_inputs(tile: np.ndarray, tile_mask: np.ndarray, tile_fg: np.ndarray
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(y0, mask, cond) for one repair tile, each with a batch dim of 1.

    tile: uint8 (P, P) damaged print; tile_mask, tile_fg: bool (P, P).
    """
    y0 = to_model_range(datasets.to_tensor(tile))[None]
    mask = datasets.to_tensor(tile_mask)[None]
    orientation = torch.from_numpy(datasets.filled_orientation(tile, tile_mask, tile_fg))[None]
    return y0, mask, torch.cat([y0, mask, orientation], dim=1)
