"""Train any experiment: picks the right trainer from its folder and params.json.

Usage:  python experiments/train.py --exp E07_aux_minutiae_cldice [--smoke] [--steps N] [--calibrate 20]
- ResShift with the plain diffusion loss (E02):       fpr/training/train_resshift.py
- ResShift with aux losses (E06-E08, E12-E16, E26):   fpr/training/train_aux.py (--calibrate first if lams are null)
- an experiment with its own train.py (E05, E09, E10, E11, E19, E20): that script
- CCDF re-samplings of trained models (E21-E25) and E01, E03, E04, E17, E18 train nothing.
Other arguments are passed on to the trainer.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

from fpr import experiments


def trainer(name: str) -> str:
    """The module or script that trains `name`."""
    own = experiments.experiment_dir(name) / "train.py"
    if own.exists():
        return str(own)
    params = experiments.load_params(name)
    if "aux_losses" in params and "training" in params:
        return "fpr.training.train_aux"
    if "diffusion" in params and "training" in params:
        return "fpr.training.train_resshift"
    raise SystemExit(f"{name} has no training step (a re-sampling of another model, or not learned)")


def main() -> None:
    args = sys.argv[1:]
    assert "--exp" in args, __doc__
    name = args[args.index("--exp") + 1]
    target = trainer(name)
    print(f"{name}: {target}", flush=True)
    if target.endswith(".py"):                     # the experiment's own script reads its folder's params.json
        rest = args[:args.index("--exp")] + args[args.index("--exp") + 2:]
        sys.argv = [target] + rest
        runpy.run_path(target, run_name="__main__")
    else:
        sys.argv = [target] + args
        runpy.run_module(target, run_name="__main__")


if __name__ == "__main__":
    main()
