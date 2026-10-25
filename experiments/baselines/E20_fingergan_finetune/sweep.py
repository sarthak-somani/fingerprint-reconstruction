"""E20 val sweep: whole print vs paste-back on v3 val, both evaluated; the winner goes to selected.json.

Usage (GPU):  FPR_DATA_VERSION=v3 python experiments/baselines/E20_fingergan_finetune/sweep.py
      --needed: only print 'yes' / 'no': does E17's val sweep call for E20 (params.json run_if)?
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: makes config/fpr importable

import numpy as np
import pandas as pd

from experiments.baselines.E17_fingergan_norm import val_sweep
from fpr import experiments
from fpr.data.normalization import normalize
from fpr.models.base import paste_back

EXP = "E20_fingergan_finetune"
HERE = Path(__file__).resolve().parent


def needed() -> bool:
    """True when E17's selected val variant still fails: its net-rescue 95% CI lies below 0."""
    table = pd.read_csv(HERE.parent / "E17_fingergan_norm" / "sweep_val.csv")
    best = json.loads((HERE.parent / "E17_fingergan_norm" / "selected.json").read_text())["label"]
    return bool(table.set_index("label").loc[best, "net_rescue_hi"] < 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--needed", action="store_true")
    args = parser.parse_args()
    if args.needed:
        print("yes" if needed() else "no")
        return
    from experiments.baselines.E20_fingergan_finetune.reconstructor import FingerGanFinetunedReconstructor
    experiments.check_data_version(EXP)
    method = FingerGanFinetunedReconstructor(experiments.load_params(EXP), selected={"paste_back": False})

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
