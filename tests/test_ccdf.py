"""The CCDF sampler (fpr/models/ccdf.py), E09's reconstructor and the v2 sweep's selection rule (no checkpoints needed)."""
import numpy as np
import pandas as pd
import torch

from experiments.main.E09_ccdf.reconstructor import build_reconstructor, E02_PARAMS
from fpr import experiments
from fpr.models import ccdf, resshift
from fpr.models.schedule import build_schedule
from fpr.models.unet import UNet
from fpr.validation import ccdf_sweep as sweep

PARAMS = experiments.load_params("E09_ccdf")
TINY = dict(E02_PARAMS["model"], base_channels=8, channel_mults=[1, 2], attention_resolutions=[])   # fast on CPU


def tiny_batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    y0 = torch.rand(2, 1, 32, 32, generator=g) * 2 - 1
    mask = (torch.rand(2, 1, 32, 32, generator=g) > 0.6).float()
    cond = torch.cat([y0, mask, torch.rand(2, 2, 32, 32, generator=g)], dim=1)
    return y0, mask, cond


def test_full_chain_from_y0_is_e02_bit_for_bit():
    model, schedule = UNet(**TINY, patch=32).eval(), build_schedule()
    y0, mask, cond = tiny_batch()
    torch.manual_seed(1)
    ours = ccdf.sample(model, y0, mask, cond, y0, schedule["steps"], False, schedule)
    torch.manual_seed(1)
    e02 = resshift.sample(model, y0, mask, cond, schedule=schedule)
    assert torch.equal(ours, e02)


def test_projection_and_t0_zero():
    model, schedule = UNet(**TINY, patch=32).eval(), build_schedule()
    y0, mask, cond = tiny_batch()
    x_init = torch.zeros_like(y0)
    assert torch.equal(ccdf.sample(model, y0, mask, cond, x_init, 0, False, schedule),
                       mask * x_init + (1 - mask) * y0)                       # t0 = 0: the estimate itself
    out = ccdf.sample(model, y0, mask, cond, x_init, 4, True, schedule)
    assert torch.equal(out[mask == 0], y0[mask == 0]) and out.abs().max() <= 1


def test_reconstructor_on_a_small_print(tmp_path):
    torch.save({"ema": UNet(**E02_PARAMS["model"]).state_dict(), "step": 0}, tmp_path / "e02.pt")
    regressor = UNet(**E02_PARAMS["model"]).eval()
    yy, xx = np.mgrid[0:200, 0:150]
    print_ = (127 + 120 * np.cos(2 * np.pi * (xx + 0.4 * yy) / 9)).astype(np.uint8)
    mask = np.zeros(print_.shape, bool)
    mask[60:120, 40:100] = True
    for cfg in ({"init": "regressor", "t0": 2, "project": True}, {"init": "e02_x0", "t0": 1, "project": False}):
        method = build_reconstructor(PARAMS, config=cfg, regressor=regressor, checkpoint=tmp_path / "e02.pt")
        first, second = method.repair(print_, mask), method.repair(print_, mask)
        assert first.shape == print_.shape and np.array_equal(first[~mask], print_[~mask])
        assert np.array_equal(first, second)                                   # fixed seed


def test_selection_rule():
    rows = [{"config": "y0_t15", "init": "y0", "t0": 15, "project": False, "eer": 0.12, "genuine_mean": 40, "impostor_p95": 16},
            {"config": "a", "init": "regressor", "t0": 2, "project": True, "eer": 0.125, "genuine_mean": 45, "impostor_p95": 16},
            {"config": "b", "init": "e02_x0", "t0": 4, "project": False, "eer": 0.10, "genuine_mean": 50, "impostor_p95": 17},
            {"config": "c", "init": "y0", "t0": 8, "project": False, "eer": 0.14, "genuine_mean": 60, "impostor_p95": 15}]
    chosen = sweep.select(pd.DataFrame(rows))
    assert (chosen["init"], chosen["t0"], chosen["project"]) == ("regressor", 2, True)   # b: impostors up; c: EER +2
    assert len(sweep.configurations(PARAMS["sweep"])) == 2 + 3 * 4 * 2
