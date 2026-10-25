"""E21_ccdf_aux_minutiae: E06_aux_minutiae's checkpoint sampled with E09's CCDF sampler.

E09_ccdf's CCDF sampler on another experiment's trained weights (params.json 'model': its checkpoint and
params.json), with the configuration chosen on v2 val by fpr/validation/ccdf_sweep.py (selected.json).
No retraining; the regression control is not used.
"""
import json
from pathlib import Path

from fpr.models.reconstructors import CcdfReconstructor, resolve

HERE = Path(__file__).resolve().parent


def build_reconstructor(params: dict, config: dict | None = None, random_init: bool = False) -> CcdfReconstructor:
    source = json.loads(resolve(params["model"]["params"], HERE).read_text())
    config = config or json.loads((HERE / params["inference"]["selected"]).read_text())
    assert config["init"] != "regressor", "no regression control here"
    checkpoint = None if random_init else resolve(params["model"]["checkpoint"], HERE)
    return CcdfReconstructor(source, params["inference"], checkpoint, config)
