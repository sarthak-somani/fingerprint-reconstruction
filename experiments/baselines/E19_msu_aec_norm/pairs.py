"""Normalised (damaged, clean, mask) training windows from the stored v3 train set, for E19 and E20.

NormPairs reuses fpr.data.datasets.FixedTrainDataset(normalized=True): the same files (script 10's
damaged_norm/ and train/clean_norm/), the same padding (grey 128) and the same window choice (centred on
damage with probability config.FIXED_PATCH_ON_DAMAGE, else anywhere on the finger). It skips the
orientation field FixedTrainDataset computes for E02-style models, which these whole-print enhancers
do not take, and returns tensors in [0, 1].
"""
import numpy as np
import torch
from torch.utils.data import get_worker_info

import config
from fpr.data import eval_sets, foreground, image_io, patches
from fpr.data.datasets import FixedTrainDataset, to_tensor


class NormPairs(FixedTrainDataset):
    """Items: (degraded, clean, mask) float32 (K, 1, P, P) in [0, 1]: K windows of one stored sample.

    Several windows per loaded sample (K = `windows`) because loading and decoding the PNGs, not the
    tiny networks, limits the training speed; each window is placed independently.
    """

    def __init__(self, patch: int, seed: int = config.TRAIN_SEED, windows: int = 1):
        super().__init__(patch=patch, seed=seed, normalized=True)
        self.windows = windows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        worker = get_worker_info()
        rng = np.random.default_rng([self.seed, self.epoch, 0 if worker is None else worker.id, index])
        clean, degraded, mask = self.load(self.meta.iloc[index])
        fg = foreground.foreground_mask(clean)
        out = []
        for _ in range(self.windows):
            aim = mask if (mask.any() and rng.random() < config.FIXED_PATCH_ON_DAMAGE) else fg
            top, left = patches.patch_corner(aim, rng, self.patch, min_fg=0.0)
            window = (slice(top, top + self.patch), slice(left, left + self.patch))
            out.append((to_tensor(degraded[window]), to_tensor(clean[window]), to_tensor(mask[window])))
        return tuple(torch.stack(t) for t in zip(*out))


def flatten(batch: torch.Tensor) -> torch.Tensor:
    """(B, K, 1, P, P) from a DataLoader over NormPairs -> (B * K, 1, P, P)."""
    return batch.flatten(0, 1)


def val_windows(n: int, patch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """n fixed (degraded, clean, mask) windows from the normalised val samples (script 10's probes_norm/).

    One window per sample (cycling over the samples), centred on its damage. For the validation loss.
    """
    meta = eval_sets.load_meta("val")
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        sample = meta.iloc[k % len(meta)]["sample_id"]
        clean, degraded = (image_io.pad_to_at_least(image_io.load_gray(eval_sets.normalized_triplet_path(
            "val", sample, kind)), patch, value=config.NORMALIZE_BACKGROUND) for kind in ("clean", "degraded"))
        mask = image_io.pad_to_at_least(image_io.load_mask(eval_sets.triplet_path("val", sample, "mask")),
                                        patch, value=False)
        top, left = patches.patch_corner(mask if mask.any() else foreground.foreground_mask(clean), rng, patch,
                                         min_fg=0.0)
        window = (slice(top, top + patch), slice(left, left + patch))
        out.append((to_tensor(degraded[window]), to_tensor(clean[window]), to_tensor(mask[window])))
    return tuple(torch.stack(t) for t in zip(*out))
