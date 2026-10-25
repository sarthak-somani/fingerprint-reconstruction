"""Ridge binarisation (numpy) and the differentiable soft skeleton (Shit et al., "clDice", CVPR 2021)
that the clDice term of fpr/training/aux_losses.py uses."""
import numpy as np
import torch
import torch.nn.functional as F
from skimage.filters import threshold_sauvola

import config


def binarize(image: np.ndarray) -> np.ndarray:
    """bool (H, W), True on ridges (dark pixels), by Sauvola local thresholding."""
    threshold = threshold_sauvola(image, window_size=config.BINARIZE_WINDOW_PX)
    return image < threshold


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    """3x3 soft min (plus-shaped), as in Shit et al. 2021. x: (B, 1, H, W)."""
    vertical = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    """3x3 soft max. x: (B, 1, H, W)."""
    return F.max_pool2d(x, 3, stride=1, padding=1)


def soft_skeleton(x: torch.Tensor, iterations: int = config.SOFT_SKELETON_ITERS) -> torch.Tensor:
    """Differentiable skeleton of a soft foreground map x in [0, 1], (B, 1, H, W).

    Iterated soft erosion; at each step keep what an opening would remove.
    """
    skeleton = F.relu(x - soft_dilate(soft_erode(x)))
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - soft_dilate(soft_erode(x)))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton
