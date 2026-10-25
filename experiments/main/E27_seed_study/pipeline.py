"""E27 after training: repair, evaluate and analyse every run whose checkpoint is ready (skips what is done).

Usage:  FPR_DATA_VERSION=v3 python experiments/main/E27_seed_study/pipeline.py post --train-seeds 1      # or 0, 2, all
        FPR_DATA_VERSION=v3 python experiments/main/E27_seed_study/pipeline.py status
Repairs run as separate sample.py processes, --parallel-sample at a time on the one GPU; evaluations as
evaluate_run.py processes, --parallel-eval at a time, each with an equal share of the job's CPUs, only on an
A40 / L40 node (NFIQ 2 needs libssl 1.1; elsewhere they are left for a later "post all" job). Then
analysis.py on whatever is evaluated, holding this folder's analysis lock. Logs: logs/<run>_{sample,eval}.out.
"""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: makes config/fpr importable

from experiments.main.E27_seed_study import runs

PY = sys.executable
HERE = runs.HERE
ANALYSIS_LOCK = HERE / ".analysis.lock"


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                              text=True, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return ""


def run_parallel(commands: dict[str, list[str]], parallel: int, env: dict | None = None) -> dict[str, int]:
    """{label: argv} -> {label: exit code}, at most `parallel` processes at once, each logging to logs/<label>.out."""
    pending, active, codes = list(commands.items()), {}, {}
    while pending or active:
        while pending and len(active) < parallel:
            label, argv = pending.pop(0)
            log = open(HERE / "logs" / f"{label}.out", "a")
            active[label] = (subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env), log, time.time())
            print(f"{time.strftime('%H:%M:%S')}  start {label}", flush=True)
        for label, (proc, log, start) in list(active.items()):
            if proc.poll() is not None:
                log.close()
                codes[label] = proc.returncode
                del active[label]
                print(f"{time.strftime('%H:%M:%S')}  {label}: exit {proc.returncode} after {(time.time() - start) / 60:.1f} min",
                      flush=True)
        time.sleep(5)
    return codes


def status(names: list[str]) -> None:
    for name in names:
        model, s, r = runs.parse(name)
        print(f"{name:<12} checkpoint {'ready' if runs.checkpoint_ready(model, s) else 'not ready':<10} "
              f"repaired {runs.repaired(name)!s:<6} evaluated {runs.evaluated(name)}")


def post(train_seeds: list[int], parallel_sample: int, parallel_eval: int) -> None:
    names = runs.all_runs(train_seeds)
    ready = [n for n in names if runs.checkpoint_ready(*runs.parse(n)[:2])]
    print(f"{len(ready)} of {len(names)} runs have a finished checkpoint", flush=True)
    todo = [n for n in ready if not runs.repaired(n)]
    if todo:
        if not gpu_name():
            print("no GPU: repairs skipped", flush=True)
        else:
            run_parallel({f"{n}_sample": [PY, str(HERE / "sample.py"), "--run", n] for n in todo}, parallel_sample)
    todo = [n for n in ready if runs.repaired(n) and not runs.evaluated(n)]
    gpu = gpu_name()
    if todo and not ("A40" in gpu or "L40" in gpu):
        print(f"evaluation needs an A40 or L40 node (NFIQ 2), this is {gpu or 'no GPU'}: {len(todo)} left for later",
              flush=True)
    elif todo:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
        env = dict(os.environ, E27_EVAL_WORKERS=str(max(2, cpus // min(parallel_eval, len(todo)))))
        run_parallel({f"{n}_eval": [PY, str(HERE / "evaluate_run.py"), "--run", n] for n in todo}, parallel_eval, env)
    status(names)
    with open(ANALYSIS_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        code = subprocess.run([PY, str(HERE / "analysis.py")]).returncode
        print(f"analysis.py exit {code}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["post", "status"])
    parser.add_argument("--train-seeds", default="all", help="0, 1, 2 or all")
    parser.add_argument("--parallel-sample", type=int, default=3)
    parser.add_argument("--parallel-eval", type=int, default=3)
    args = parser.parse_args()
    seeds = list(runs.TRAIN_SEEDS) if args.train_seeds == "all" else [int(args.train_seeds)]
    (HERE / "logs").mkdir(exist_ok=True)
    if args.stage == "status":
        status(runs.all_runs(seeds))
    else:
        post(seeds, args.parallel_sample, args.parallel_eval)


if __name__ == "__main__":
    main()
