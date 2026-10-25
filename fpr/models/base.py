"""Reconstructor interface, full-image tiling and paste-back.

Every method (identity, gabor, the diffusion models, the baselines) is a
Reconstructor. Scripts call `repair()`, which always ends in `paste_back`, so
pixels outside the damage mask can never change.
"""
from typing import Callable

import numpy as np
from scipy import ndimage

import config


def paste_back(original: np.ndarray, reconstructed: np.ndarray, mask: np.ndarray,
               feather_px: int = config.FEATHER_PX) -> np.ndarray:
    """Keep `original` outside `mask`; use `reconstructed` inside, feathered at the seam.

    Inputs: uint8 (H, W) original (the damaged input), uint8 (H, W)
    reconstruction, bool (H, W) mask. The blend weight rises from 0 at the
    mask edge to 1 at `feather_px` pixels inside it, so every pixel outside
    the mask is bit-identical to `original`. Output: uint8 (H, W).
    """
    inside_distance = ndimage.distance_transform_edt(mask)        # 0 outside the mask
    weight = np.clip(inside_distance / max(feather_px, 1), 0.0, 1.0)
    blended = weight * reconstructed.astype(np.float32) + (1 - weight) * original
    out = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    out[~mask] = original[~mask]                                  # exact, not just ~equal
    return out


def tile_window(patch: int, overlap: int) -> np.ndarray:
    """float32 (patch, patch) blending window: raised-cosine ramps of width `overlap`."""
    ramp = np.ones(patch, np.float32)
    if overlap > 0:
        rise = 0.5 - 0.5 * np.cos(np.pi * (np.arange(overlap) + 0.5) / overlap)
        ramp[:overlap], ramp[-overlap:] = rise, rise[::-1]
    return np.maximum(np.outer(ramp, ramp), 1e-3)


def tile_starts(length: int, patch: int, stride: int) -> list[int]:
    """Start offsets so tiles of size `patch` cover [0, length) (last one flush with the end)."""
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch, stride))
    return starts + [length - patch]


def reconstruct_full_image(model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                           degraded: np.ndarray, mask: np.ndarray,
                           patch: int = config.TILE_PATCH,
                           overlap: int = config.TILE_OVERLAP) -> np.ndarray:
    """Run a patch-level model over a full print and blend the tiles.

    `model_fn(patch_image uint8 (p, p), patch_mask bool (p, p)) -> uint8 (p, p)`
    is called only on tiles that contain damage. Overlapping outputs are
    blended with `tile_window`; pixels no tile covered keep `degraded`.
    Images smaller than `patch` are padded with white. Output: uint8 (H, W).
    Call paste_back afterwards (Reconstructor.repair does).
    """
    h, w = degraded.shape
    ph, pw = max(h, patch), max(w, patch)
    image = np.full((ph, pw), config.BACKGROUND_VALUE, np.uint8)
    image[:h, :w] = degraded
    padded_mask = np.zeros((ph, pw), bool)
    padded_mask[:h, :w] = mask

    window = tile_window(patch, overlap)
    total = np.zeros((ph, pw), np.float32)
    weight = np.zeros((ph, pw), np.float32)
    for top in tile_starts(ph, patch, patch - overlap):
        for left in tile_starts(pw, patch, patch - overlap):
            tile_mask = padded_mask[top:top + patch, left:left + patch]
            if not tile_mask.any():
                continue
            out = model_fn(image[top:top + patch, left:left + patch], tile_mask)
            total[top:top + patch, left:left + patch] += window * out
            weight[top:top + patch, left:left + patch] += window

    covered = weight > 0
    result = image.astype(np.float32)
    result[covered] = total[covered] / weight[covered]
    return np.clip(np.round(result[:h, :w]), 0, 255).astype(np.uint8)


def reconstruct_full_image_batched(batch_fn: Callable[[np.ndarray, np.ndarray, list], np.ndarray],
                                   degraded: np.ndarray, mask: np.ndarray,
                                   patch: int = config.TILE_PATCH,
                                   overlap: int = config.TILE_OVERLAP,
                                   background: int = config.BACKGROUND_VALUE) -> np.ndarray:
    """Like reconstruct_full_image, but all damaged tiles go to the model in one call.

    `batch_fn(tiles uint8 (N, p, p), tile_masks bool (N, p, p), corners [(top, left)] * N)
    -> uint8 (N, p, p)`, so a GPU model can sample every tile of a print together.
    Corners refer to the image padded to at least `patch` with `background` (white; normalised
    prints use config.NORMALIZE_BACKGROUND). Output: uint8 (H, W).
    """
    h, w = degraded.shape
    ph, pw = max(h, patch), max(w, patch)
    image = np.full((ph, pw), background, np.uint8)
    image[:h, :w] = degraded
    padded_mask = np.zeros((ph, pw), bool)
    padded_mask[:h, :w] = mask

    corners = [(top, left) for top in tile_starts(ph, patch, patch - overlap)
               for left in tile_starts(pw, patch, patch - overlap)
               if padded_mask[top:top + patch, left:left + patch].any()]
    if not corners:
        return degraded.copy()
    tiles = np.stack([image[t:t + patch, l:l + patch] for t, l in corners])
    tile_masks = np.stack([padded_mask[t:t + patch, l:l + patch] for t, l in corners])
    outputs = batch_fn(tiles, tile_masks, corners)

    window = tile_window(patch, overlap)
    total = np.zeros((ph, pw), np.float32)
    weight = np.zeros((ph, pw), np.float32)
    for (top, left), out in zip(corners, outputs):
        total[top:top + patch, left:left + patch] += window * out
        weight[top:top + patch, left:left + patch] += window
    covered = weight > 0
    result = image.astype(np.float32)
    result[covered] = total[covered] / weight[covered]
    return np.clip(np.round(result[:h, :w]), 0, 255).astype(np.uint8)


class Reconstructor:
    """Base class. Subclasses set `name` and implement `reconstruct`.

    reconstruct(degraded uint8 (H, W), mask bool (H, W)) -> uint8 (H, W)
    num_parameters() -> int, 0 for non-learned methods.
    """
    name = "base"

    def reconstruct(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def num_parameters(self) -> int:
        return 0

    def repair(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """reconstruct() followed by paste_back(): the only entry point scripts use."""
        return paste_back(degraded, self.reconstruct(degraded, mask), mask)
