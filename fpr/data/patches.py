"""Sample fixed-size training patches that lie on the finger.

Used by fpr/data/datasets.py: TrainPatchDataset cuts a patch and then damages it;
FixedTrainDataset cuts the same window from a stored clean/degraded/mask triplet.
"""
import numpy as np

import config
from fpr.data import image_io


def patch_corner(fg_mask: np.ndarray, rng: np.random.Generator, size: int = config.PATCH_SIZE,
                 min_fg: float = config.PATCH_MIN_FG) -> tuple[int, int]:
    """(top, left) of a random `size` x `size` window centred on a foreground pixel.

    Input: bool (H, W) foreground mask with H, W >= size (pad first).
    Tries up to config.PATCH_MAX_TRIES centres and returns the first window that
    is at least `min_fg` foreground, else the most-foreground window seen.
    """
    rows, cols = np.nonzero(fg_mask)
    if len(rows) == 0:                       # no foreground: take any window
        rows, cols = np.array([size // 2]), np.array([size // 2])

    best = None
    for _ in range(config.PATCH_MAX_TRIES):
        k = rng.integers(len(rows))
        top = int(np.clip(rows[k] - size // 2, 0, fg_mask.shape[0] - size))
        left = int(np.clip(cols[k] - size // 2, 0, fg_mask.shape[1] - size))
        fraction = fg_mask[top:top + size, left:left + size].mean()
        if best is None or fraction > best[0]:
            best = (fraction, top, left)
        if fraction >= min_fg:
            break
    return best[1], best[2]


def sample_patch(image: np.ndarray, fg_mask: np.ndarray, rng: np.random.Generator,
                 size: int = config.PATCH_SIZE, min_fg: float = config.PATCH_MIN_FG
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Random `size` x `size` crop centred on a foreground pixel.

    Input: uint8 (H, W) image, bool (H, W) foreground mask, numpy Generator.
    Output: (patch uint8 (size, size), patch_fg bool (size, size)).
    Images smaller than `size` are padded with white (background).
    """
    image = image_io.pad_to_at_least(image, size)
    fg_mask = image_io.pad_to_at_least(fg_mask, size, value=False)
    top, left = patch_corner(fg_mask, rng, size, min_fg)
    return (image[top:top + size, left:left + size].copy(),
            fg_mask[top:top + size, left:left + size].copy())
