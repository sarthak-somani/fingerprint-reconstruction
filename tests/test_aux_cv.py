"""Tests for E08's terms in fpr/training/aux_losses.py: the Laplacian edge term and the MS-SSIM / L1 mix.

Real train patches: ~0 on (clean, clean), large on (damaged, clean); finite gradients that reach
x0_hat only; single-scale SSIM equal to skimage's.
"""
import pytest
import torch
from skimage.metrics import structural_similarity

import config
from fpr.data import datasets
from fpr.training import aux_losses

needs_train_set = pytest.mark.skipif(not config.TRAIN_META_CSV.exists(), reason="run script 08 first")


def real_patches(n: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(clean, degraded, m): n damaged train patches in [0, 1] and their masks dilated by 4 px."""
    data = datasets.FixedTrainDataset()
    items = [data[i] for i in range(0, len(data), len(data) // (2 * n))]
    items = [item for item in items if item["mask"].sum() > 0][:n]
    clean, degraded, mask = (torch.stack([item[k] for item in items]) for k in ("clean", "degraded", "mask"))
    return clean, degraded, aux_losses.dilate(mask, 4)


def test_laplacian_kernel():
    x = torch.zeros(1, 1, 5, 5)
    x[0, 0, 2, 2] = 1
    lap = aux_losses._laplacian(x)[0, 0]
    assert lap[2, 2] == -4 and lap[1, 2] == lap[2, 1] == lap[3, 2] == lap[2, 3] == 1 and lap.sum() == 0


@needs_train_set
def test_terms_on_real_patches():
    """~0 when the prediction is the clean patch, clearly larger when it is the damaged one."""
    clean, degraded, m = real_patches()
    damaged = aux_losses.composite(degraded, clean, m)
    edge_same, edge_damaged = aux_losses.cv_edge_term(clean, clean, m), aux_losses.cv_edge_term(damaged, clean, m)
    mix_same, mix_damaged = aux_losses.cv_mix_term(clean, clean, m), aux_losses.cv_mix_term(damaged, clean, m)
    assert edge_same.shape == mix_same.shape == (len(clean),)
    assert edge_same.max() < 1.01e-3                                    # the Charbonnier floor sqrt(1e-6)
    assert mix_same.abs().max() < 1e-5
    assert edge_damaged.mean() > 0.05, f"edge term on damage: {edge_damaged.mean():.4f}"
    assert mix_damaged.mean() > 0.1, f"mix term on damage: {mix_damaged.mean():.4f}"


@needs_train_set
def test_gradients_are_finite_and_reach_x0_hat_only():
    """Inputs slightly outside [0, 1], an identical sample and a border-only mask included."""
    clean, degraded, m = real_patches(4)
    x0_hat = (degraded * 1.1 - 0.05).requires_grad_()                 # range [-0.05, 1.05]
    x0, m = clean.clone().requires_grad_(), m.clone()
    m[2] = 0
    m[2, :, :6, :] = 1                                                  # empty at scale 4 after the crop
    x0_hat.data[3] = clean[3]                                           # xc = x0 for sample 3
    for term in (aux_losses.cv_edge_term, aux_losses.cv_mix_term):
        x0_hat.grad, x0.grad = None, None
        xc = aux_losses.composite(x0_hat, x0, m)
        values = term(xc, x0, m)
        values.sum().backward()
        assert torch.isfinite(values).all() and torch.isfinite(x0_hat.grad).all()
        assert x0_hat.grad[m > 0].abs().sum() > 0 and x0_hat.grad[m == 0].eq(0).all()
        assert x0.grad is None or x0.grad.eq(0).all()                   # x0 is a target, not a parameter
        assert x0_hat.grad[2].abs().sum() > 0                           # the border sample still learns


@needs_train_set
def test_single_scale_ssim_matches_skimage():
    """Valid-region SSIM with an all-ones mask equals skimage's mean SSIM (which crops 5 px too)."""
    clean, degraded, _ = real_patches(3)
    for x, y in zip(degraded, clean):
        ssim_map, _ = aux_losses._ssim_maps(x[None], y[None])
        ours = aux_losses.masked_mean(ssim_map, torch.ones_like(ssim_map)).item()
        theirs = structural_similarity(x[0].double().numpy(), y[0].double().numpy(), gaussian_weights=True,
                                       sigma=1.5, use_sample_covariance=False, data_range=1)
        assert ours == pytest.approx(theirs, abs=1e-3)


def test_no_random_numbers_used():
    xc, x0, m = torch.rand(2, 1, 128, 128), torch.rand(2, 1, 128, 128), torch.ones(2, 1, 128, 128)
    before = torch.get_rng_state()
    aux_losses.cv_edge_term(xc, x0, m), aux_losses.cv_mix_term(xc, x0, m)
    assert torch.equal(before, torch.get_rng_state())
