"""Tests for fpr/models/base.py: paste-back guarantee and full-image tiling."""
import numpy as np

from fpr.models.base import paste_back, reconstruct_full_image, Reconstructor
from fpr.models.registry import get_reconstructor


def random_case(shape=(200, 170), seed=0):
    rng = np.random.default_rng(seed)
    original = rng.integers(0, 256, shape, dtype=np.uint8)
    reconstructed = rng.integers(0, 256, shape, dtype=np.uint8)
    mask = np.zeros(shape, bool)
    mask[50:120, 40:100] = True
    mask[150:160, 10:160] = True
    return original, reconstructed, mask


def test_outside_mask_unchanged():
    original, reconstructed, mask = random_case()
    out = paste_back(original, reconstructed, mask)
    assert out.dtype == np.uint8
    assert np.array_equal(out[~mask], original[~mask])


def test_deep_inside_mask_is_reconstruction():
    original, reconstructed, mask = random_case()
    out = paste_back(original, reconstructed, mask, feather_px=3)
    assert np.array_equal(out[60:110, 50:90], reconstructed[60:110, 50:90])


class Scrambler(Reconstructor):
    """A deliberately bad method that changes every pixel."""
    name = "scrambler"

    def reconstruct(self, degraded, mask):
        return 255 - degraded


def test_repair_always_pastes_back():
    original, _, mask = random_case()
    for method in (Scrambler(), get_reconstructor("identity")):
        out = method.repair(original, mask)
        assert np.array_equal(out[~mask], original[~mask])
    assert np.array_equal(get_reconstructor("identity").repair(original, mask), original)


def test_tiling_identity_model_is_lossless():
    original, _, mask = random_case(shape=(300, 260))
    out = reconstruct_full_image(lambda image, m: image, original, mask, patch=128, overlap=32)
    assert np.array_equal(out, original)


def test_tiling_covers_mask_and_small_images():
    original, _, mask = random_case(shape=(100, 90))     # smaller than one patch
    out = reconstruct_full_image(lambda image, m: np.zeros_like(image), original, mask)
    assert out.shape == original.shape
    assert (out[mask] == 0).all()


def test_batched_tiling_matches_per_tile_tiling():
    from fpr.models.base import reconstruct_full_image_batched
    original, _, mask = random_case(shape=(300, 260))
    darken = lambda image, m: (image // 2).astype(np.uint8)
    per_tile = reconstruct_full_image(darken, original, mask)
    batched = reconstruct_full_image_batched(lambda tiles, masks, corners: tiles // 2, original, mask)
    assert np.array_equal(per_tile, batched)
