"""The ResShift U-Net, its inputs and ResShiftReconstructor (no trained checkpoint needed)."""
import copy

import numpy as np
import torch

from fpr import experiments
from fpr.models import inputs, resshift
from fpr.models.reconstructors import ResShiftReconstructor
from fpr.models.unet import UNet

PARAMS = experiments.load_params("E02_resshift")


def test_unet_shape_and_size():
    model = UNet(**PARAMS["model"])
    out = model(torch.randn(2, 1, 128, 128), torch.tensor([1, 15]), torch.randn(2, 4, 128, 128))
    assert out.shape == (2, 1, 128, 128)
    assert 7e6 < sum(p.numel() for p in model.parameters()) < 9e6      # "lightweight", ~8M


def test_one_training_step_is_finite():
    model = UNet(**dict(PARAMS["model"], base_channels=8, channel_mults=[1, 2], attention_resolutions=[]))
    batch = {"clean": torch.rand(2, 1, 128, 128), "degraded": torch.rand(2, 1, 128, 128),
             "mask": (torch.rand(2, 1, 128, 128) > 0.7).float(), "orientation": torch.rand(2, 2, 128, 128)}
    x0, y0, mask, cond = inputs.batch_inputs(batch, torch.device("cpu"))
    assert x0.min() >= -1 and cond.shape == (2, 4, 128, 128)
    loss = resshift.training_loss(model, x0, y0, mask, cond)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters()
                                        if p.grad is not None)


def test_reconstructor_on_a_small_print(tmp_path):
    params = copy.deepcopy(PARAMS)
    params["inference"].update(sampling_steps=2, tile_batch=8)
    torch.save({"ema": UNet(**params["model"]).state_dict(), "step": 0}, tmp_path / "untrained.pt")
    method = ResShiftReconstructor(params, params["inference"], tmp_path / "untrained.pt")

    yy, xx = np.mgrid[0:200, 0:150]
    print_ = (127 + 120 * np.cos(2 * np.pi * (xx + 0.4 * yy) / 9)).astype(np.uint8)
    mask = np.zeros(print_.shape, bool)
    mask[60:120, 40:100] = True
    first, second = method.repair(print_, mask), method.repair(print_, mask)
    assert first.shape == print_.shape and first.dtype == np.uint8
    assert np.array_equal(first[~mask], print_[~mask])                 # paste-back guarantee
    assert np.array_equal(first, second)                               # fixed sampling seed
