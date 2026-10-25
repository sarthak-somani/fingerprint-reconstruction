"""Tests for fpr/training/aux_losses.py: the shared helpers, the gate and the minutiae term (E06).

The clDice (E07) and edge / MS-SSIM (E08) terms have their own test files.
"""
import pytest
import torch

import config
from fpr.data import datasets
from fpr.training import aux_losses

AUX = {"dilate_px": 4, "eta_max": 0.5, "lam_m": 4.0, "cldice_tau": 0.05, "cldice_iters": 4}
needs_maps = pytest.mark.skipif(not (config.TRAIN_MINUTIAE_DIR / "meta.csv").exists(),
                                reason="run scripts 08 and 09 first")


def square_masks(batch: int = 3, size: int = 32) -> torch.Tensor:
    """(batch, 1, size, size) masks with a 6 x 6 damaged square in the middle."""
    mask = torch.zeros(batch, 1, size, size)
    mask[:, :, 13:19, 13:19] = 1
    return mask


def test_dilate_grows_by_px():
    grown = aux_losses.dilate(square_masks(1), 4)
    assert grown.sum() == (6 + 2 * 4) ** 2 and grown.max() == 1


def test_composite_is_x0_outside_the_mask():
    x0_hat, x0, m = torch.rand(2, 1, 32, 32), torch.rand(2, 1, 32, 32), square_masks(2)
    xc = aux_losses.composite(x0_hat, x0, m)
    assert torch.equal(xc[m == 0], x0[m == 0]) and torch.equal(xc[m == 1], x0_hat[m == 1])


def test_masked_mean_per_sample():
    values, m = torch.ones(2, 1, 8, 8), torch.zeros(2, 1, 8, 8)
    values[1] *= 3
    m[0, 0, :2], m[1, 0, :4] = 1, 1
    assert torch.equal(aux_losses.masked_mean(values, m), torch.tensor([1.0, 3.0]))
    assert aux_losses.masked_mean(values, torch.zeros_like(m)).eq(0).all()     # empty mask -> 0


def test_gradients_reach_x0_hat_only():
    x0_hat = (torch.rand(3, 1, 32, 32) * 2 - 1).requires_grad_()
    x0 = (torch.rand(3, 1, 32, 32) * 2 - 1).requires_grad_()
    mask, w_norm = square_masks().requires_grad_(), torch.rand(3, 1, 32, 32).requires_grad_()
    total, _ = aux_losses.aux_loss(x0_hat, x0, mask, torch.zeros(3), {"minutiae": 1.0}, AUX, w_norm)
    total.backward()
    assert x0_hat.grad.abs().sum() > 0
    assert x0.grad is None and mask.grad is None and w_norm.grad is None
    grown = aux_losses.dilate(square_masks(), AUX["dilate_px"])
    assert x0_hat.grad[grown == 0].eq(0).all()                          # nothing outside the mask


def test_no_random_numbers_used():
    x0_hat, x0, w_norm = torch.rand(3, 1, 32, 32), torch.rand(3, 1, 32, 32), torch.rand(3, 1, 32, 32)
    before = torch.get_rng_state()
    aux_losses.aux_loss(x0_hat, x0, square_masks(), torch.zeros(3), {"minutiae": 1.0}, AUX, w_norm)
    assert torch.equal(before, torch.get_rng_state())


def test_gate_picks_low_noise_samples_with_damage():
    mask = square_masks(4)
    mask[3] = 0                                                         # no damage at all
    gate = aux_losses.eta_gate(torch.tensor([0.1, 0.5, 0.6, 0.1]), aux_losses.dilate(mask, 4), 0.5)
    assert gate.tolist() == [True, True, False, False]


def test_loss_is_zero_when_no_sample_passes():
    x0_hat = torch.rand(3, 1, 32, 32, requires_grad=True)
    total, logs = aux_losses.aux_loss(x0_hat, torch.rand(3, 1, 32, 32), square_masks(),
                                      torch.full((3,), 0.9), {"minutiae": 2.0}, AUX, torch.rand(3, 1, 32, 32))
    assert total.item() == 0 and logs["n_gated"].item() == 0 and logs["minutiae"].item() == 0


def test_loss_is_lam_times_term():
    args = torch.rand(3, 1, 32, 32), torch.rand(3, 1, 32, 32), square_masks(), torch.zeros(3)
    total, logs = aux_losses.aux_loss(*args, {"minutiae": 2.5}, AUX, torch.rand(3, 1, 32, 32))
    assert total.item() == pytest.approx(2.5 * logs["minutiae"].item())
    assert logs["weighted_minutiae"].item() == pytest.approx(total.item())


@needs_maps
def test_minutiae_term_on_real_patches():
    """~0 when the prediction is the clean patch, clearly larger when it is the damaged one."""
    data = datasets.FixedTrainDataset(minutiae_dir=config.TRAIN_MINUTIAE_DIR)
    items = [data[i] for i in range(0, len(data), len(data) // 8)]
    items = [item for item in items if item["mask"].sum() > 0]
    clean, degraded, mask, w_norm = (torch.stack([item[k] for item in items])
                                     for k in ("clean", "degraded", "mask", "minutiae_w"))
    clean, degraded = clean * 2 - 1, degraded * 2 - 1                  # the model's range, [-1, 1]
    eta = torch.zeros(len(items))
    same, _ = aux_losses.aux_loss(clean, clean, mask, eta, {"minutiae": 1.0}, AUX, w_norm)
    damaged, _ = aux_losses.aux_loss(degraded, clean, mask, eta, {"minutiae": 1.0}, AUX, w_norm)
    assert same.item() < 1e-6
    assert damaged.item() > 0.05, f"damaged patches should cost clearly more, got {damaged.item():.4f}"
    assert w_norm.max() > 0                                             # the maps hold minutiae
