"""The E27 grid: which checkpoint and sampler configuration each run uses, and where its files go.

A run is <model>_s<training seed>_r<sampler seed>, e.g. E22_s1_r2: model E09 (E02_resshift's plain-loss weights)
or E22 (E07_aux_minutiae_cldice's minutiae + clDice weights), sampled with that model's FIXED CCDF configuration
(its selected.json, chosen once on val for training seed 0: one-call x0 start, t0 = 2, projection on).
"""
import json
import re
from pathlib import Path

from fpr.experiments import experiment_dir

HERE = Path(__file__).resolve().parent
MODELS = {   # model -> (trainer whose weights it samples, experiment whose sampler configuration it uses)
    "E09": ("E02_resshift", "E09_ccdf"),
    "E22": ("E07_aux_minutiae_cldice", "E22_ccdf_aux_minutiae_cldice"),
}
TRAIN_SEEDS = (0, 1, 2)
SAMPLER_SEEDS = (0, 1, 2)
FINAL_STEP = 100_000
SPLIT = "test"
DATA_VERSION = "v3"
NAME = re.compile(r"^(E09|E22)_s(\d+)_r(\d+)$")
TIME_FILE = "timing.csv"          # written last by sample.py: a repair set is complete when it exists


def run_name(model: str, train_seed: int, sampler_seed: int) -> str:
    return f"{model}_s{train_seed}_r{sampler_seed}"


def parse(name: str) -> tuple[str, int, int]:
    match = NAME.match(name)
    assert match, f"run names look like E22_s1_r2, got {name!r}"
    return match[1], int(match[2]), int(match[3])


def all_runs(train_seeds=TRAIN_SEEDS, sampler_seeds=SAMPLER_SEEDS) -> list[str]:
    return [run_name(m, s, r) for m in MODELS for s in train_seeds for r in sampler_seeds]


def checkpoint(model: str, train_seed: int) -> Path:
    """Seed 0: the original E02 / E07 weights (exactly what E09 / E22 sample); else E27's retrain."""
    source = MODELS[model][0]
    if train_seed == 0:
        return experiment_dir(source) / "checkpoints" / "last.pt"
    return HERE / "checkpoints" / f"{source}_s{train_seed}" / "last.pt"


def checkpoint_ready(model: str, train_seed: int) -> bool:
    """The training run reached its final step (train.py saves step_100000.pt with last.pt)."""
    return (checkpoint(model, train_seed).parent / f"step_{FINAL_STEP:06d}.pt").exists()


def sampler_config(model: str) -> dict:
    """The fixed CCDF configuration of the model's experiment (selected.json, chosen on val at seed 0)."""
    selected = json.loads((experiment_dir(MODELS[model][1]) / "selected.json").read_text())
    return {k: selected[k] for k in ("init", "t0", "project")}


def benchmark_json(model: str) -> Path:
    """The cost file run_eval.py would pass for this model's experiment (shown in the report only)."""
    return experiment_dir(MODELS[model][1]) / "benchmark.json"


def recon_dir(name: str, root: Path = HERE) -> Path:
    return root / "reconstructions" / DATA_VERSION / SPLIT / name


def eval_dir(name: str, root: Path = HERE) -> Path:
    return root / "eval" / name


def repaired(name: str, root: Path = HERE) -> bool:
    return (recon_dir(name, root) / TIME_FILE).exists()


def evaluated(name: str, root: Path = HERE) -> bool:
    return (eval_dir(name, root) / "probes.csv").exists() and (eval_dir(name, root) / "headline.csv").exists()
