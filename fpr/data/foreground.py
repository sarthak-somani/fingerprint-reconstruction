"""Block-variance foreground mask: where on the image the ridges are.

Used by scripts/03 (crop the training pool), fpr/data/patches.py (keep patches on
the finger) and fpr/degradations (damage only lands on the finger).
"""
import numpy as np
from scipy import ndimage

import config


def block_std(image: np.ndarray, block: int) -> np.ndarray:
    """Standard deviation of each non-overlapping `block` x `block` tile.

    Input: uint8 (H, W). Output: float32 (ceil(H/block), ceil(W/block)),
    computed on the image scaled to [0, 1]. Edge tiles are padded with white.
    """
    h, w = image.shape
    rows, cols = -(-h // block), -(-w // block)
    padded = np.full((rows * block, cols * block), config.BACKGROUND_VALUE, np.float32)
    padded[:h, :w] = image
    tiles = padded.reshape(rows, block, cols, block) / 255.0
    return tiles.std(axis=(1, 3)).astype(np.float32)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 4-connected True region of a bool mask."""
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    return labels == (1 + int(np.argmax(sizes)))


def disk(radius: int) -> np.ndarray:
    """Bool disk structuring element of the given radius (in mask cells)."""
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return xx ** 2 + yy ** 2 <= radius ** 2


def cut_joint_strip(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove the strip of the next finger joint below the crease (rolled prints).

    The crease is a bright horizontal band. Within the lower part of the mask
    we find the row whose mean darkness (inside the mask) is lowest; if it is
    much lighter than a typical ridge row, every mask row below it is dropped.
    Input/output: bool (H, W) masks; `image` is uint8 (H, W).
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if len(rows) < 2:
        return mask
    darkness = ((255.0 - image) * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
    darkness = ndimage.uniform_filter1d(darkness, config.JOINT_SMOOTH_PX)
    start = rows[0] + int(config.JOINT_SEARCH_FROM * (rows[-1] - rows[0]))
    search = np.arange(start, rows[-1] + 1)
    crease = search[np.argmin(darkness[search])]
    if darkness[crease] < config.JOINT_DARKNESS_RATIO * np.median(darkness[rows]):
        mask = mask.copy()
        mask[crease:] = False
    return mask


def foreground_mask(image: np.ndarray, block: int = config.FG_BLOCK,
                    rolled: bool = False) -> np.ndarray:
    """Bool (H, W) mask, True on the fingerprint, False on background.

    Steps: block std above a per-image threshold (a fraction of the high
    percentile of block std, so faint prints still work), closing (fill gaps
    between ridges), hole filling, opening (cut the thin bridge to the joint
    strip on rolled prints), keep the largest component, upsample to pixels.
    With rolled=True the joint strip below the crease is also cut off.
    """
    stds = block_std(image, block)
    reference = np.percentile(stds, config.FG_REFERENCE_PERCENTILE)
    threshold = max(config.FG_MIN_STD, config.FG_RELATIVE_THRESHOLD * reference)
    blocks = stds > threshold
    blocks = np.pad(blocks, config.FG_CLOSE_BLOCKS)            # room for closing at edges
    blocks = ndimage.binary_closing(blocks, disk(config.FG_CLOSE_BLOCKS))
    blocks = blocks[config.FG_CLOSE_BLOCKS:-config.FG_CLOSE_BLOCKS,
                    config.FG_CLOSE_BLOCKS:-config.FG_CLOSE_BLOCKS]
    blocks = ndimage.binary_fill_holes(blocks)
    blocks = ndimage.binary_opening(blocks, disk(config.FG_OPEN_BLOCKS))
    blocks = largest_component(blocks)
    pixels = np.kron(blocks, np.ones((block, block), bool))
    pixels = pixels[:image.shape[0], :image.shape[1]]
    if rolled:
        pixels = largest_component(cut_joint_strip(image, pixels))
    return pixels
