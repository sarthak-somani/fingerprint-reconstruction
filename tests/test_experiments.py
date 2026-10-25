"""Tests for fpr/experiments.py and the evaluate.py file layout (fpr/data/eval_sets.py)."""
import numpy as np
import pytest

import config
from fpr import experiments
from fpr.data import eval_sets
from fpr.models.gabor import GaborReconstructor


def test_naming_rule():
    assert experiments.NAME_PATTERN.match("E01_gabor")
    assert experiments.NAME_PATTERN.match("E12_diffusion_small")
    for bad in ("gabor", "E1_gabor", "E01-Gabor", "_template"):
        assert not experiments.NAME_PATTERN.match(bad)
    assert "_template" not in experiments.list_experiments()


def test_builtin_method_from_params():
    assert isinstance(experiments.build_reconstructor("E01_gabor"), GaborReconstructor)


def test_every_experiment_is_found_in_its_group():
    names = experiments.list_experiments()
    assert {"E02_resshift", "E04_msu_stft", "E09_ccdf", "E22_ccdf_aux_minutiae_cldice"} <= set(names)
    assert experiments.experiment_dir("E09_ccdf").parent.name == "main"
    assert experiments.experiment_dir("E04_msu_stft").parent.name == "baselines"


def test_file_names_match_evaluate_py():
    assert eval_sets.layout_name("00002312_01", "oil", "degraded") == "00002312_01_oil.png"
    assert eval_sets.layout_name("00002312_01", "oil", "mask") == "00002312_01_oil_mask.png"
    path = experiments.reconstruction_path("E01_gabor", "test", "00002312_01", "oil")
    assert path.name == "00002312_01_oil_rec.png" and path.parent.name == "test"


@pytest.mark.skipif(not (config.DATA_DIR / "test" / config.LAYOUT_SUBDIR).exists(),
                    reason="run preprocessing script 07 first")
def test_eval_layout_complete():
    for split in config.EVAL_SPLITS:
        n = len(eval_sets.load_meta(split))
        for part in eval_sets.LAYOUT_PARTS:
            assert len(list(eval_sets.layout_dir(split, part).glob("*.png"))) == n
