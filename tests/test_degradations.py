"""Tests for fpr/degradations and fpr/patches.

Checks the degrade() contract (dtypes, shapes, damage only inside the mask
and the foreground), determinism under a fixed seed, and patch sampling.
"""
import numpy as np
import pytest

from fpr.data import patches
from fpr.data.degradations import apply_degradation, apply_random_degradation, CONTAMINANTS


def synthetic_print(size: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """Sinusoidal ridges (period 9 px) inside a centred disk of foreground."""
    yy, xx = np.mgrid[0:size, 0:size]
    ridges = (127.5 + 127.5 * np.cos(2 * np.pi * (xx + 0.3 * yy) / 9)).astype(np.uint8)
    fg = np.hypot(yy - size / 2, xx - size / 2) < size * 0.4
    ridges[~fg] = 255
    return ridges, fg


@pytest.mark.parametrize("name", CONTAMINANTS)
@pytest.mark.parametrize("severity", [0.0, 0.5, 1.0])
def test_degrade_contract(name, severity):
    image, fg = synthetic_print()
    degraded, mask = apply_degradation(image, fg, name, severity, np.random.default_rng(0))
    assert degraded.dtype == np.uint8 and degraded.shape == image.shape
    assert mask.dtype == bool and mask.shape == image.shape
    assert not (mask & ~fg).any(), "damage outside the foreground"
    assert np.array_equal(degraded[~mask], image[~mask]), "pixels changed outside the mask"
    assert mask.any(), "no damage at all"


@pytest.mark.parametrize("name", CONTAMINANTS)
def test_same_seed_same_damage(name):
    image, fg = synthetic_print()
    a = apply_degradation(image, fg, name, 0.7, np.random.default_rng(3))
    b = apply_degradation(image, fg, name, 0.7, np.random.default_rng(3))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_random_degradation_info():
    image, fg = synthetic_print()
    _, _, info = apply_random_degradation(image, fg, np.random.default_rng(1), names=["oil"])
    assert info["contaminant"] == "oil" and 0.0 <= info["severity"] <= 1.0


def test_sample_patch_on_foreground():
    image, fg = synthetic_print(300)
    patch, patch_fg = patches.sample_patch(image, fg, np.random.default_rng(0), size=128)
    assert patch.shape == (128, 128) and patch.dtype == np.uint8
    assert patch_fg.mean() >= 0.7


@pytest.mark.parametrize("name", CONTAMINANTS)
def test_no_blank_white_patches(name):
    """At full severity no 16x16 block inside the damage turns flat white (ridges wiped out)."""
    image, fg = synthetic_print(240)
    degraded, mask = apply_degradation(image, fg, name, 1.0, np.random.default_rng(5))
    for r in range(0, 240 - 16, 16):
        for c in range(0, 240 - 16, 16):
            if mask[r:r + 16, c:c + 16].all():
                block = degraded[r:r + 16, c:c + 16]
                assert not (block.mean() > 245 and block.std() < 5), f"blank white block at {r},{c}"


def test_viscous_oil_is_stronger():
    from fpr.data.degradations import oil
    image, fg = synthetic_print(240)
    strong, mask_s = oil.degrade(image, fg, 1.0, np.random.default_rng(2), viscous=True)
    mild, mask_m = oil.degrade(image, fg, 1.0, np.random.default_rng(2), viscous=False)
    assert np.array_equal(mask_s, mask_m)   # same rng draws -> same film region
    change = lambda out: np.abs(out.astype(int) - image)[mask_s].mean()
    assert change(strong) > change(mild) > 0
