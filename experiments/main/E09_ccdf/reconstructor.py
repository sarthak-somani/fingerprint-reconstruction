"""E09_ccdf: E02_resshift's trained network, sampled CCDF-style (fpr/models/ccdf.py).

The starting estimate, t0 and projection come from selected.json, written by train.py's val sweep.
"""
import json
from pathlib import Path

import torch

import config
from fpr.models.reconstructors import CcdfReconstructor, resolve
from fpr.models.unet import UNet

HERE = Path(__file__).resolve().parent
E02_PARAMS = json.loads(config.REFERENCE_PARAMS.read_text())


def load_regressor(path: Path, device: str) -> UNet:
    """The regression control's EMA weights (train.py), on E02's U-Net."""
    net = UNet(**E02_PARAMS["model"]).to(device).eval()
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True)["ema"])
    return net


def build_reconstructor(params: dict, config: dict | None = None, regressor: torch.nn.Module | None = None,
                        checkpoint: Path | None = None, random_init: bool = False) -> CcdfReconstructor:
    config = config or json.loads((HERE / params["inference"]["selected"]).read_text())
    checkpoint = None if random_init else checkpoint or resolve(params["e02"]["checkpoint"], HERE)
    method = CcdfReconstructor(E02_PARAMS, params["inference"], checkpoint, config, regressor)
    if regressor is None and config["init"] == "regressor":
        method.regressor = load_regressor(HERE / "checkpoints" / "regressor_last.pt", method.device)
    return method
