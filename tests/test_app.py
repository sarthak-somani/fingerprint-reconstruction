"""The demo app (demo/): model list, damage, evaluator-consistent scoring, and one end-to-end run.

Uses the real test set and checkpoints; tests skip what is not there. The end-to-end run uses E04
(no network, CPU), plus E09 on a GPU node (it takes ~20 s per print on CPU).
"""
import numpy as np
import pytest
import torch

import config
from demo import damage, handlers, models, scoring
from fpr import experiments
from fpr.data import eval_sets

needs_test_set = pytest.mark.skipif(not eval_sets.meta_path(config.DEMO_SPLIT).exists(),
                                    reason="no demo split on disk")


def test_only_loadable_models_are_offered():
    offered = models.available_models()
    assert all(not models.missing_files(name) for name in offered)
    e12 = "E12_norm_ccdf_cldice_cv"
    if e12 in experiments.list_experiments():
        has_selection = (experiments.experiment_dir(e12) / "selected.json").exists()
        assert (e12 in offered) == (has_selection and not models.missing_files(e12))
        assert models.output_normalized(e12)
    assert not models.output_normalized("E09_ccdf")


def test_verdict_is_strictly_above_tau():
    tau = scoring.tau()
    if tau is None:
        pytest.skip("no evaluation of this data version yet")
    assert scoring.verdict(tau) == "NO MATCH"           # evaluate.py accepts score > tau
    assert scoring.verdict(tau + 1) == "MATCH"


def test_ridge_fidelity_of_identical_prints_is_one():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (64, 64)).astype(np.uint8)
    region = np.zeros((64, 64), bool)
    region[16:48, 16:48] = True
    assert scoring.ridge_fidelity(image, image, region) == pytest.approx(1.0)
    assert scoring.psnr(image, image, region) == scoring.evaluator().PSNR_CAP_DB


@needs_test_set
def test_stacked_contaminants_are_reproducible_and_union_their_masks():
    clean, _ = damage.load_sample(damage.sample_ids()[0])
    one, mask_one = damage.stacked_contaminants(clean, ["oil"], 0.6, seed=3)
    two, mask_two = damage.stacked_contaminants(clean, ["oil", "scar"], 0.6, seed=3)
    again, mask_again = damage.stacked_contaminants(clean, ["oil", "scar"], 0.6, seed=3)
    assert np.array_equal(two, again) and np.array_equal(mask_two, mask_again)
    assert mask_one.any() and (mask_two | mask_one).sum() == mask_two.sum()   # the first mask is kept


def test_drawn_smudge_changes_only_painted_pixels():
    clean = np.full((40, 30), 200, np.uint8)
    layer = np.zeros((40, 30, 4), np.uint8)
    layer[10:20, 5:15] = (32, 32, 32, 255)
    composite = np.repeat(clean[..., None], 4, axis=2)
    composite[10:20, 5:15, :3] = 32
    degraded, mask = damage.drawn_smudge({"background": clean, "layers": [layer], "composite": composite}, clean)
    assert mask.sum() == 100 and (degraded[mask] == 32).all() and (degraded[~mask] == 200).all()


@needs_test_set
@pytest.mark.parametrize("source", damage.MASK_SOURCES)
def test_run_demo_end_to_end(source):
    if source == damage.PREDICTED_MASK and not (config.REPO_ROOT / "segmenter/checkpoints/best.pt").exists():
        pytest.skip("no segmenter checkpoint")
    offered = models.available_models()
    if "E04_msu_stft" not in offered:
        pytest.skip("E04 not available")
    second = "E09_ccdf" if "E09_ccdf" in offered and torch.cuda.is_available() else handlers.NO_MODEL   # ~20 s on CPU
    sample = damage.sample_ids()[0]
    out = handlers.run_demo(sample, None, None, damage.TEST_DAMAGE, [], 0.6, 0, source, "E04_msu_stft", second)
    clean = out[0]
    assert len(out) == 8 and isinstance(out[-1], str)
    assert out[1].shape == clean.shape + (3,) and out[3].shape == clean.shape
    assert "E04_msu_stft" in out[-1] and "damaged, no repair" in out[-1]
    if scoring.nbis_available() and scoring.tau() is not None:
        assert "MATCH" in out[-1]


@needs_test_set
def test_upload_cannot_use_test_set_damage():
    with pytest.raises(ValueError):
        handlers.run_demo(None, np.full((64, 64), 255, np.uint8), None, damage.TEST_DAMAGE, [], 0.6, 0,
                          damage.TRUE_MASK, "E04_msu_stft", handlers.NO_MODEL)
