"""PyTorch datasets for the team's model.

TrainPatchDataset: random 128 x 128 patches from train/clean/, damaged on the fly.
FixedTrainDataset: 128 x 128 patches from the stored damaged train set (train/damaged/).
EvalSetDataset: the fixed full-size val/test probes from disk.
Tensors: images float32 (1, H, W) in [0, 1]; mask float32 (1, H, W) in {0, 1}.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info

import config
from fpr.data import eval_sets, foreground, image_io, orientation, patches, split
from fpr.data.degradations import apply_random_degradation


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """uint8 (H, W) image or bool (H, W) mask -> float32 (1, H, W) in [0, 1]."""
    scale = 1.0 if image.dtype == bool else 255.0
    return torch.from_numpy(image.astype(np.float32) / scale)[None]


def filled_orientation(degraded: np.ndarray, mask: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """2-channel (cos 2theta, sin 2theta) float32 (2, H, W) from the degraded image,
    measured outside the damage and filled into it (fpr.data.orientation.fill_orientation)."""
    theta, _ = orientation.orientation_field(degraded)
    theta = orientation.fill_orientation(theta, fg & ~mask)
    return orientation.doubled_angle(theta)


class TrainPatchDataset(Dataset):
    """Each item: a random pool print -> random patch -> fresh random damage.

    Returns dict: clean, degraded, mask (1, P, P); orientation (2, P, P);
    contaminant (str); severity (float). The rng for item i is built from
    (seed, epoch, worker id, i), so workers never share random streams and
    results are reproducible. Call set_epoch(e) each epoch for new damage.
    """

    def __init__(self, pool_dir: Path = config.TRAIN_CLEAN_DIR, epoch_size: int = config.EPOCH_SIZE,
                 patch: int = config.PATCH_SIZE, seed: int = config.TRAIN_SEED,
                 contaminants: list[str] | None = None):
        self.pool_dir = Path(pool_dir)
        self.pool = pd.read_csv(self.pool_dir / "pool.csv", dtype={"participant_id": str})
        if split.split_available():            # not copied to the GPU cluster? then trust script 03
            leaked = set(self.pool["participant_id"]) - set(split.participants_in("train"))
            assert not leaked, f"training pool contains non-train participants: {sorted(leaked)[:5]}"
        self.epoch_size, self.patch, self.seed = epoch_size, patch, seed
        self.contaminants = contaminants
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.epoch_size

    def item_rng(self, index: int) -> np.random.Generator:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        return np.random.default_rng([self.seed, self.epoch, worker_id, index])

    def __getitem__(self, index: int) -> dict:
        rng = self.item_rng(index)
        row = self.pool.iloc[rng.integers(len(self.pool))]
        image = image_io.load_gray(self.pool_dir / row["file"])
        fg = foreground.foreground_mask(image)
        clean, patch_fg = patches.sample_patch(image, fg, rng, size=self.patch)
        degraded, mask, info = apply_random_degradation(clean, patch_fg, rng, self.contaminants)
        return {"clean": to_tensor(clean), "degraded": to_tensor(degraded), "mask": to_tensor(mask),
                "orientation": torch.from_numpy(filled_orientation(degraded, mask, patch_fg)),
                "contaminant": info["contaminant"], "severity": info["severity"]}


class FixedTrainDataset(Dataset):
    """The stored damaged train set (preprocessing/08_make_train_set.py).

    Each item: one stored sample (a clean crop + one of its damaged versions, one per contaminant)
    -> a 128 x 128 window cut at the same place from clean, degraded and mask.
    With probability config.FIXED_PATCH_ON_DAMAGE the window is centred on a damaged
    pixel (else anywhere on the finger), so most patches contain damage. Same return
    dict as TrainPatchDataset, plus sample_id. Damage is fixed; only the window varies
    with (seed, epoch, worker id, index).
    With minutiae_dir (e.g. config.TRAIN_MINUTIAE_DIR, script 09), each item also has
    "minutiae_w": the clean crop's minutiae map float32 (1, P, P) in [0, 1], cut with the same
    window. The window does not depend on it.
    With normalized=True, clean and degraded come from the MSU-normalised copies (script 10;
    train/clean_norm/, <train set>/damaged_norm/) and pad with config.NORMALIZE_BACKGROUND.
    The window is chosen on the normalised clean crop, so items differ from normalized=False.
    """

    def __init__(self, train_dir: Path = config.TRAIN_SET_DIR, patch: int = config.PATCH_SIZE,
                 seed: int = config.TRAIN_SEED, contaminants: list[str] | None = None,
                 minutiae_dir: Path | None = None, normalized: bool = False):
        self.train_dir, self.normalized = Path(train_dir), normalized
        self.minutiae_dir = None if minutiae_dir is None else Path(minutiae_dir)
        meta = pd.read_csv(self.train_dir / "meta.csv", dtype={"participant_id": str})
        if contaminants:
            meta = meta[meta["contaminant"].isin(contaminants)]
        self.meta = meta.reset_index(drop=True)
        if split.split_available():
            leaked = set(self.meta["participant_id"]) - set(split.participants_in("train"))
            assert not leaked, f"train set contains non-train participants: {sorted(leaked)[:5]}"
        self.patch, self.seed, self.epoch = patch, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.meta)

    def load(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(clean, degraded, mask), padded to at least one patch."""
        damaged = self.train_dir / "damaged" / row["sample_id"]
        if self.normalized:
            clean = image_io.load_gray(config.TRAIN_CLEAN_NORM_DIR / row["clean_file"])
            degraded = image_io.load_gray(self.train_dir / "damaged_norm" / f"{row['sample_id']}_degraded.png")
        else:
            clean = image_io.load_gray(config.TRAIN_CLEAN_DIR / row["clean_file"])
            degraded = image_io.load_gray(f"{damaged}_degraded.png")
        mask = image_io.load_mask(f"{damaged}_mask.png")
        background = config.NORMALIZE_BACKGROUND if self.normalized else config.BACKGROUND_VALUE
        return (image_io.pad_to_at_least(clean, self.patch, value=background),
                image_io.pad_to_at_least(degraded, self.patch, value=background),
                image_io.pad_to_at_least(mask, self.patch, value=False))

    def __getitem__(self, index: int) -> dict:
        worker = get_worker_info()
        rng = np.random.default_rng([self.seed, self.epoch, 0 if worker is None else worker.id, index])
        row = self.meta.iloc[index]
        clean, degraded, mask = self.load(row)
        fg = foreground.foreground_mask(clean)
        aim = mask if (mask.any() and rng.random() < config.FIXED_PATCH_ON_DAMAGE) else fg
        top, left = patches.patch_corner(aim, rng, self.patch, min_fg=0.0)
        window = (slice(top, top + self.patch), slice(left, left + self.patch))
        # The orientation input uses the damaged print's finger mask, as at repair time,
        # where the clean print is not available.
        fg = foreground.foreground_mask(degraded)[window]
        clean, degraded, mask = clean[window], degraded[window], mask[window]
        item = {"clean": to_tensor(clean), "degraded": to_tensor(degraded), "mask": to_tensor(mask),
                "orientation": torch.from_numpy(filled_orientation(degraded, mask, fg)),
                "contaminant": row["contaminant"], "severity": float(row["severity"]),
                "sample_id": row["sample_id"]}
        if self.minutiae_dir is not None:
            w_norm = self.load_minutiae(row)[window]
            item["minutiae_w"] = torch.from_numpy(w_norm.astype(np.float32))[None]
        return item

    def load_minutiae(self, row: pd.Series) -> np.ndarray:
        """The clean crop's float16 (H, W) minutiae map, padded like the crop (with 0)."""
        w_norm = np.load(self.minutiae_dir / Path(row["clean_file"]).with_suffix(".npy"))
        return image_io.pad_to_at_least(w_norm, self.patch, value=0)


class EvalSetDataset(Dataset):
    """Fixed full-size probes of a split ('val' or 'test') from preprocessing/data/{split}/probes/.

    Returns dict: clean, degraded, mask (1, H, W); probe_id, contaminant.
    Sizes differ between prints, so use batch_size=1.
    """

    def __init__(self, split: str):
        self.split = split
        self.meta = eval_sets.load_meta(split)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> dict:
        row = self.meta.iloc[index]
        load = {kind: image_io.load_gray(eval_sets.triplet_path(self.split, row["sample_id"], kind))
                for kind in ("clean", "degraded", "mask")}
        return {"clean": to_tensor(load["clean"]), "degraded": to_tensor(load["degraded"]),
                "mask": to_tensor(load["mask"] > 127), "probe_id": row["probe_id"],
                "contaminant": row["contaminant"]}
