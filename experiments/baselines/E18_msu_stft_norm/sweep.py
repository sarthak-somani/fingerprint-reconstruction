"""E18 val check: the same STFT repairs of v3 val measured as stored vs normalised again (E04's protocol).

Usage:  FPR_DATA_VERSION=v3 python experiments/baselines/E18_msu_stft_norm/sweep.py
Writes sweep_val.csv and selected.json ({"label": ...}); the test run is worth doing only for 'as_stored'.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: makes config/fpr importable

from experiments.baselines.E17_fingergan_norm import val_sweep
from experiments.baselines.E18_msu_stft_norm.reconstructor import MsuStftNormReconstructor
from fpr import experiments

EXP = "E18_msu_stft_norm"
HERE = Path(__file__).resolve().parent


def main() -> None:
    experiments.check_data_version(EXP)
    params = experiments.load_params(EXP)
    flags = params["sweep"]["variants"]
    method = MsuStftNormReconstructor(params)
    repairs = lambda degraded, mask: dict.fromkeys(flags, method.repair(degraded, mask))
    table = val_sweep.sweep(EXP, repairs, list(flags), rec_normalized=flags)
    best = val_sweep.choose(table)
    (HERE / "selected.json").write_text(json.dumps({"label": best}, indent=2) + "\n")
    print(f"selected {best}")


if __name__ == "__main__":
    main()
