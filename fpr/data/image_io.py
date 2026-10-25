"""Load and save fingerprint PNGs, pad them and crop them.

Used everywhere an image touches disk. Images are uint8 (H, W) grayscale,
dark ridges on a white (255) background; masks are bool (H, W).
"""
from pathlib import Path

import cv2
import numpy as np

import config


def load_gray(path: str | Path) -> np.ndarray:
    """Read a PNG as uint8 (H, W) grayscale in [0, 255]."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def load_mask(path: str | Path) -> np.ndarray:
    """Read a mask PNG saved by `save_png` as bool (H, W); True where pixel > 127."""
    return load_gray(path) > 127


def save_png(path: str | Path, image: np.ndarray) -> None:
    """Write a uint8 (H, W) image or a bool mask (saved as 0/255) to PNG.

    Parent folders are created. Refuses to write inside the raw NIST folder.
    """
    path = Path(path)
    # Resolve both sides: raw_data/ holds symlinks, so compare the real folders.
    raw_folders = {config.RAW_DATA_SOURCE.resolve(), config.RAW_DATA_DIR.resolve()}
    assert not raw_folders & set(path.resolve().parents), "never write into raw data"
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    assert image.dtype == np.uint8 and image.ndim == 2, "expected uint8 (H, W)"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def pad_to_at_least(image: np.ndarray, size: int, value: int = config.BACKGROUND_VALUE
                    ) -> np.ndarray:
    """Pad bottom/right so both sides are >= `size`. Same dtype as input."""
    pad_h = max(0, size - image.shape[0])
    pad_w = max(0, size - image.shape[1])
    return np.pad(image, ((0, pad_h), (0, pad_w)), constant_values=value)


def bounding_box(mask: np.ndarray, margin: int = 0) -> tuple[int, int, int, int]:
    """(row0, row1, col0, col1) of the True pixels, grown by `margin`, clipped.

    row1/col1 are exclusive, so image[row0:row1, col0:col1] is the crop.
    An empty mask gives the full image.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    row0 = max(0, rows[0] - margin)
    row1 = min(mask.shape[0], rows[-1] + 1 + margin)
    col0 = max(0, cols[0] - margin)
    col1 = min(mask.shape[1], cols[-1] + 1 + margin)
    return row0, row1, col0, col1


def crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop with a box from `bounding_box`; works for images and masks."""
    row0, row1, col0, col1 = box
    return image[row0:row1, col0:col1]
