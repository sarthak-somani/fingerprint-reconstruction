"""Image-quality metrics restricted to the damaged region.

Secondary to matching, but reported per probe by fpr/validation/quick_check.py and
shown in the demo. All take uint8 (H, W) images and a bool (H, W) region.
"""
from functools import lru_cache

import numpy as np
import torch
from skimage.metrics import structural_similarity

import config
from fpr.data import image_io, orientation


def masked_psnr(pred: np.ndarray, clean: np.ndarray, region: np.ndarray) -> float:
    """PSNR in dB over `region` pixels (peak 255); capped at config.PSNR_CAP_DB."""
    if not region.any():
        return float("nan")
    mse = np.mean((pred[region].astype(np.float64) - clean[region]) ** 2)
    if mse == 0:
        return config.PSNR_CAP_DB
    return float(min(config.PSNR_CAP_DB, 10 * np.log10(255.0 ** 2 / mse)))


def masked_ssim(pred: np.ndarray, clean: np.ndarray, region: np.ndarray) -> float:
    """Mean of the full SSIM map (7x7 windows) over `region`; in [-1, 1]."""
    if not region.any():
        return float("nan")
    _, ssim_map = structural_similarity(pred, clean, data_range=255, full=True)
    return float(ssim_map[region].mean())


@lru_cache(maxsize=1)
def lpips_model() -> torch.nn.Module:
    """The LPIPS network, loaded once (CPU, eval mode)."""
    import lpips
    return lpips.LPIPS(net=config.LPIPS_NET, verbose=False).eval()


def region_box(region: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of `region` plus a margin, grown to at least MIN_BOX_SIDE_PX per side."""
    row0, row1, col0, col1 = image_io.bounding_box(region, margin=config.METRIC_BOX_MARGIN_PX)
    h, w = region.shape
    grow_r = max(0, config.MIN_BOX_SIDE_PX - (row1 - row0))
    grow_c = max(0, config.MIN_BOX_SIDE_PX - (col1 - col0))
    row0, row1 = max(0, row0 - grow_r // 2), min(h, row1 + grow_r - grow_r // 2)
    col0, col1 = max(0, col0 - grow_c // 2), min(w, col1 + grow_c - grow_c // 2)
    return row0, row1, col0, col1


def lpips_box(pred: np.ndarray, clean: np.ndarray, region: np.ndarray) -> float:
    """LPIPS distance (lower = closer, ~0-1) on the damaged-region bounding box."""
    if not region.any():
        return float("nan")
    box = region_box(region)

    def to_tensor(image: np.ndarray) -> torch.Tensor:
        crop = image_io.crop(image, box).astype(np.float32) / 127.5 - 1.0   # [-1, 1]
        return torch.from_numpy(crop)[None, None].repeat(1, 3, 1, 1)

    with torch.no_grad():
        return float(lpips_model()(to_tensor(pred), to_tensor(clean)).item())


def orientation_error(pred: np.ndarray, clean: np.ndarray, region: np.ndarray) -> float:
    """Mean ridge-orientation difference in degrees over `region` (0 = same ridge flow)."""
    theta_pred, _ = orientation.orientation_field(pred)
    theta_clean, _ = orientation.orientation_field(clean)
    return orientation.orientation_error_deg(theta_pred, theta_clean, region)


def all_image_metrics(pred: np.ndarray, clean: np.ndarray, region: np.ndarray) -> dict:
    """Dict with psnr, ssim, lpips, orientation_error_deg for one probe."""
    return {"psnr": masked_psnr(pred, clean, region),
            "ssim": masked_ssim(pred, clean, region),
            "lpips": lpips_box(pred, clean, region),
            "orientation_error_deg": orientation_error(pred, clean, region)}
