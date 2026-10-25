"""Retrain E02_resshift or E07_aux_minutiae_cldice with another training seed, nothing else changed.

Usage (on a GPU, via chain.sh train):  python experiments/main/E27_seed_study/train.py --source E02_resshift --seed 1
CPU check:   python experiments/main/E27_seed_study/train.py --source E07_aux_minutiae_cldice --seed 1 --smoke --out <dir>
Runs the source experiment's own trainer (fpr/training/train_resshift.py for E02, train_aux.py for E07) with its
params.json, except training.seed. The seed sets the network init, t and noise draws, the loader's shuffling
and the patch windows (FixedTrainDataset(seed=)). The fixed validation patches and their t / noise stay at
seed 0, so val.csv is comparable with the seed-0 run. Writes to checkpoints/<source>_s<seed>/ (resumes from
last.pt) and seed_info.json there.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: makes config/fpr importable

import config
from fpr import experiments
from fpr.training import common, train_aux, train_resshift

HERE = Path(__file__).resolve().parent
SOURCES = ("E02_resshift", "E07_aux_minutiae_cldice")   # the models behind E09_ccdf and E22_ccdf_aux_minutiae_cldice
VAL_SEED = 0                                             # the seed-0 runs' validation patches and draws


def override_seed(params: dict, seed: int) -> dict:
    """A deep copy of params with training.seed = seed and nothing else changed."""
    out = copy.deepcopy(params)
    out["training"]["seed"] = seed
    return out


def checkpoint_dir(source: str, seed: int) -> Path:
    return HERE / "checkpoints" / f"{source}_s{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", type=Path, default=None, help="default checkpoints/<source>_s<seed>/")
    parser.add_argument("--smoke", action="store_true", help="the trainers' 20-step CPU check")
    parser.add_argument("--steps", type=int, default=None, help="a short check")
    args = parser.parse_args()
    assert config.DATA_VERSION == "v2", "E02 and E07 were trained on v2: run with FPR_DATA_VERSION unset or v2"
    assert args.seed != 0 or args.smoke or args.steps, "seed 0 is the original run (E02 / E07 checkpoints)"

    params = override_seed(experiments.load_params(args.source), args.seed)
    train = dict(params["training"])
    if args.steps:
        train.update(steps=args.steps, checkpoint_every=args.steps)
    if args.smoke:
        train.update(common.SMOKE)
    out = args.out or checkpoint_dir(args.source, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    info = {"source": args.source, "params_json": str(experiments.experiment_dir(args.source) / "params.json"),
            "training_seed": args.seed, "val_seed": VAL_SEED, "data_version": config.DATA_VERSION}
    (out / "seed_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)
    if args.source == "E02_resshift":
        train_resshift.train_model(params, train, out, val_seed=VAL_SEED)
    else:
        calibration = experiments.experiment_dir(args.source) / "calibration.json"
        train_aux.train_model(params, train, out, calibration, val_seed=VAL_SEED)


if __name__ == "__main__":
    main()
