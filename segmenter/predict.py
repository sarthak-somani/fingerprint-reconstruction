"""Run a SegFormer on a whole grey print: ImageNet-style input, logits upsampled back to the print size.

Used by evaluate_zero_shot.py (ADE20K classes, zero-shot) and, after fine-tuning, by DamageSegmenter
(one 'damage' logit -> bool mask) for the reconstructors and the demo.
"""
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalise(grey01: torch.Tensor) -> torch.Tensor:
    """float (B, 1, H, W) in [0, 1] -> (B, 3, H, W): grey copied to RGB, normalised as SegFormer was trained."""
    mean, std = IMAGENET_MEAN.to(grey01.device), IMAGENET_STD.to(grey01.device)
    return (grey01.expand(-1, 3, -1, -1) - mean) / std


def to_input(image: np.ndarray, device) -> torch.Tensor:
    """uint8 (H, W) grey print -> float (1, 3, H, W) model input on `device`."""
    return normalise(torch.from_numpy(image).float().div(255).view(1, 1, *image.shape).to(device))


@torch.no_grad()
def full_logits(model: torch.nn.Module, image: np.ndarray) -> torch.Tensor:
    """uint8 (H, W) -> float (C, H, W) logits at full resolution (bilinear from the model's 1/4 output)."""
    device = next(model.parameters()).device
    logits = model(to_input(image, device))
    return F.interpolate(logits, size=image.shape, mode="bilinear", align_corners=False)[0]


class DamageSegmenter:
    """Fine-tuned binary SegFormer: uint8 (H, W) damaged print -> bool (H, W) predicted damage mask."""

    def __init__(self, model: torch.nn.Module, threshold: float = 0.5):
        self.model, self.threshold = model.eval(), threshold

    def predict(self, image: np.ndarray) -> np.ndarray:
        return (torch.sigmoid(full_logits(self.model, image)[0]) > self.threshold).cpu().numpy()
