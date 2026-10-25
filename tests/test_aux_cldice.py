"""Tests for fpr.training.aux_losses.cldice_term (E07): ridge polarity, values and gradients on real train patches.

The shared helpers and the gate are tested in test_aux_losses.py.
"""
import numpy as np
import pytest
import torch

import config
from fpr.data import datasets
from fpr.training import aux_losses, skeleton

AUX = {"dilate_px": 4, "eta_max": 0.5, "lam_m": 4.0, "cldice_tau": 0.05, "cldice_iters": 4}
pytestmark = pytest.mark.skipif(not config.TRAIN_META_CSV.exists(), reason="run script 08 first")


@pytest.fixture(scope="module")
def patches() -> dict:
    """8 real train patches with damage, spread over the set: clean, degraded, dilated mask m, all in [0, 1]."""
    data = datasets.FixedTrainDataset()
    items = [data[i] for i in range(0, len(data), len(data) // 8)]
    items = [item for item in items if item["mask"].sum() > 0]
    batch = {k: torch.stack([item[k] for item in items]) for k in ("clean", "degraded", "mask")}
    batch["m"] = aux_losses.dilate(batch["mask"], AUX["dilate_px"])
    return batch


def test_ridge_map_is_high_on_dark_ridges(patches, monkeypatch):
    """The soft ridge maps the term builds are ~1 on the clean print's (dark) Sauvola ridges."""
    seen = []
    real_soft_skeleton = skeleton.soft_skeleton

    def recording_soft_skeleton(v, iterations):
        seen.append(v.detach())
        return real_soft_skeleton(v, iterations=iterations)

    monkeypatch.setattr(skeleton, "soft_skeleton", recording_soft_skeleton)
    clean = patches["clean"]
    aux_losses.cldice_term(clean, clean, patches["m"])
    v_l = seen[1]                                                       # V_P = r(xc), then V_L = r(x0)
    for k in range(len(clean)):
        ridges = skeleton.binarize((clean[k, 0].numpy() * 255).astype(np.uint8))
        on_ridges, on_valleys = v_l[k, 0][ridges].mean().item(), v_l[k, 0][~ridges].mean().item()
        assert on_ridges > 0.85 and on_ridges > on_valleys + 0.4, (k, on_ridges, on_valleys)


def test_small_on_clean_larger_on_damaged_largest_without_ridges(patches):
    """Not exactly 0 on (clean, clean): V_L is soft, so sum(S V) / sum(S) < 1 (measured mean 0.07)."""
    clean, m = patches["clean"], patches["m"]
    same = aux_losses.cldice_term(clean, clean, m)
    damaged = aux_losses.cldice_term(aux_losses.composite(patches["degraded"], clean, m), clean, m)
    flat = aux_losses.cldice_term(aux_losses.composite(torch.full_like(clean, 0.5), clean, m), clean, m)
    assert same.mean() < 0.1 and (same < 0.25).all()
    assert (damaged > same).all() and damaged.mean() > 1.5 * same.mean()
    assert (flat > damaged).all() and flat.mean() > 0.3                 # ridges gone in M: topology lost


def test_gradients_finite_and_only_inside_the_mask(patches):
    clean, mask = patches["clean"] * 2 - 1, patches["mask"]             # the model's range, [-1, 1]
    x0_hat = (patches["degraded"] * 2 - 1).requires_grad_()
    x0 = clean.clone().requires_grad_()
    before = torch.get_rng_state()
    total, logs = aux_losses.aux_loss(x0_hat, x0, mask, torch.zeros(len(clean)), {"cldice": 1.0}, AUX)
    total.backward()
    assert torch.equal(before, torch.get_rng_state())                  # no torch random numbers
    assert torch.isfinite(total) and logs["cldice"] > 0
    assert torch.isfinite(x0_hat.grad).all() and x0_hat.grad.abs().sum() > 0 and x0.grad is None
    assert x0_hat.grad[patches["m"] == 0].eq(0).all()                  # xc = x0 outside M
