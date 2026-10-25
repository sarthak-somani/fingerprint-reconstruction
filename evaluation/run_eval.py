"""Evaluate one experiment with the main evaluator (evaluation/evaluate.py) and file the results.

Usage: python evaluation/run_eval.py --exp E01_gabor --split test
Runs evaluate.py on preprocessing/data/{split}/layout/ and the experiment's reconstructions
(with experiments/<group>/<exp>/benchmark.json as the cost, when experiments/benchmark.py has written it, and
--rec-normalized when the experiment's params.json says its outputs are MSU-normalised prints),
then moves its timestamped output folder to evaluation/results/<exp>/<split>_<timestamp>/ and adds
a row to LEADERBOARD.md.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import config
from fpr import experiments, leaderboard
from fpr.data import eval_sets

TIMESTAMP = re.compile(r"^\d{8}_\d{6}$")              # evaluate.py's output folder name


def timestamp_folders() -> set[str]:
    """Names of evaluate.py's not-yet-filed output folders in evaluation/results/."""
    if not config.EVAL_RESULTS_DIR.exists():
        return set()
    return {p.name for p in config.EVAL_RESULTS_DIR.iterdir() if p.is_dir() and TIMESTAMP.match(p.name)}


def run_evaluator(exp: str, split: str) -> Path:
    """Run evaluate.py for one experiment and split; return its output folder."""
    recon_dir = experiments.reconstructions_dir(exp, split)
    assert any(recon_dir.glob("*_rec.png")), f"no reconstructions in {recon_dir}; run run_experiment.py"
    command = [sys.executable, str(config.EVALUATE_SCRIPT),
               str(eval_sets.layout_dir(split, "original")),
               str(eval_sets.layout_dir(split, "degraded")),
               str(recon_dir),
               str(eval_sets.layout_dir(split, "mask")),
               "--gallery", str(eval_sets.layout_dir(split, "gallery"))]
    benchmark = experiments.experiment_dir(exp) / "benchmark.json"
    if benchmark.exists():
        command += ["--benchmark", str(benchmark)]
    if experiments.load_params(exp).get("output_normalized"):
        command.append("--rec-normalized")
    before = timestamp_folders()
    subprocess.run(command, check=True)
    new = sorted(timestamp_folders() - before)
    assert len(new) == 1, f"expected one new results folder, found {new}"
    return config.EVAL_RESULTS_DIR / new[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, choices=experiments.list_experiments())
    parser.add_argument("--split", default="test", choices=config.EVAL_SPLITS)
    args = parser.parse_args()
    experiments.check_data_version(args.exp)

    produced = run_evaluator(args.exp, args.split)
    filed = config.EVAL_RESULTS_DIR / args.exp / f"{args.split}_{produced.name}"
    filed.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), filed)
    leaderboard.add_run(args.exp, args.split, filed)
    print(f"filed results in {filed}")
    print(f"updated {config.LEADERBOARD_MD}")


if __name__ == "__main__":
    main()
