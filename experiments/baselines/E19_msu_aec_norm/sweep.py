"""E19 val sweep: whole print vs paste-back on v3 val, both evaluated; the winner goes to selected.json.

Usage (GPU):  FPR_DATA_VERSION=v3 python experiments/baselines/E19_msu_aec_norm/sweep.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: makes config/fpr importable

import numpy as np

from experiments.baselines.E17_fingergan_norm import val_sweep
from experiments.baselines.E19_msu_aec_norm.reconstructor import MsuAecNormReconstructor
from fpr import experiments
from fpr.data.normalization import normalize
from fpr.models.base import paste_back

EXP = "E19_msu_aec_norm"
HERE = Path(__file__).resolve().parent


def main() -> None:
    experiments.check_data_version(EXP)
    params = experiments.load_params(EXP)
    method = MsuAecNormReconstructor(params, selected={"paste_back": False})

    def variants_fn(degraded: np.ndarray, mask: np.ndarray) -> dict:
        normalized = normalize(degraded)
        whole = method.enhance_normalized(normalized)
        return {"whole": whole, "paste_back": paste_back(normalized, whole, mask)}

    table = val_sweep.sweep(EXP, variants_fn, ["whole", "paste_back"])
    best = val_sweep.choose(table)
    (HERE / "selected.json").write_text(json.dumps({"label": best, "paste_back": best == "paste_back"},
                                                   indent=2) + "\n")
    print(f"selected {best}")


if __name__ == "__main__":
    main()
