"""E16_norm_no_aux: E12 with the plain diffusion loss only (its own sweep's configuration).

Normalise the damaged print -> CCDF-sample the damaged tiles -> paste back in the normalised domain
(fpr/models/reconstructors.py, NormalizedCcdfReconstructor). The configuration is selected.json.
"""
import json
from pathlib import Path

from fpr.models.reconstructors import NormalizedCcdfReconstructor, resolve

HERE = Path(__file__).resolve().parent


def build_reconstructor(params: dict, config: dict | None = None,
                        random_init: bool = False) -> NormalizedCcdfReconstructor:
    config = config or json.loads((HERE / params["inference"]["selected"]).read_text())
    checkpoint = None if random_init else resolve(params["inference"]["checkpoint"], HERE)
    return NormalizedCcdfReconstructor(params, params["inference"], checkpoint, config)
