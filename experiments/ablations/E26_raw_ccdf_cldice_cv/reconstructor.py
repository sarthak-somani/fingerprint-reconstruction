"""E26_raw_ccdf_cldice_cv: E12 without the MSU normalisation. CCDF-sampled on raw prints, raw output with
E02's paste-back (the evaluator normalises it like every raw-output model)."""
import json
from pathlib import Path

from fpr.models.reconstructors import CcdfReconstructor, resolve

HERE = Path(__file__).resolve().parent


def build_reconstructor(params: dict, config: dict | None = None, random_init: bool = False) -> CcdfReconstructor:
    config = config or json.loads((HERE / params["inference"]["selected"]).read_text())
    checkpoint = None if random_init else resolve(params["inference"]["checkpoint"], HERE)
    return CcdfReconstructor(params, params["inference"], checkpoint, config)
