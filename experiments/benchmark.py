"""Measure each method's repair cost: seconds per print, peak GPU memory, peak RAM.

Usage: python experiments/benchmark.py [--exp E02_resshift E04_msu_stft ...] [--n 50]
       (default: every experiment with test reconstructions for config.DATA_VERSION; run it through
       scripts/slurm/benchmark.sh so every method is timed on the same node type)
Each experiment runs in a fresh subprocess, so its peak RAM is its own. It repairs N fixed test
probes one at a time (method.repair, as run_experiment.py does) after WARMUP calls on the first
probe. Writes experiments/<group>/<exp>/benchmark.json (read by evaluation/run_eval.py) and
experiments/benchmark.md.
"""
import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import numpy as np
import pandas as pd

import config
from fpr import experiments
from fpr.data import data_md, eval_sets, image_io

WARMUP = 3
SKIP = {"E01_gabor"}            # no reconstructions for the current data version


def cpu_name() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "cpu"


def measure(exp: str, n: int) -> dict:
    """Child process: time `exp`'s repair on n test probes; peak memory of this process."""
    import torch

    meta = eval_sets.load_meta("test").sample(n=n, random_state=config.SPLIT_SEED)
    inputs = [(image_io.load_gray(eval_sets.triplet_path("test", r.sample_id, "degraded")),
               image_io.load_mask(eval_sets.triplet_path("test", r.sample_id, "mask"))) for r in meta.itertuples()]
    method = experiments.build_reconstructor(exp)
    device = str(getattr(method, "device", "cpu"))
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    for _ in range(WARMUP):
        method.repair(*inputs[0])
    sync()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    seconds = []
    for degraded, mask in inputs:
        start = time.perf_counter()
        method.repair(degraded, mask)
        sync()
        seconds.append(time.perf_counter() - start)
    return {"experiment": exp, "device": "cuda" if cuda else "cpu",
            "device_name": torch.cuda.get_device_name() if cuda else f"{cpu_name()}, {torch.get_num_threads()} threads",
            "n_prints": n, "warmup": WARMUP,
            "seconds_median": float(np.median(seconds)), "seconds_p90": float(np.percentile(seconds, 90)),
            "seconds_mean": float(np.mean(seconds)),
            "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else None,
            "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,   # Linux: KiB
            "parameters": int(method.num_parameters()),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "host": platform.node()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", nargs="*", default=None, help="default: every experiment with test reconstructions")
    parser.add_argument("--n", type=int, default=50, help="test probes to time")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(measure(args.child, args.n)))
        return

    exps = args.exp or [e for e in experiments.list_experiments() if e not in SKIP
                        and any(experiments.reconstructions_dir(e, "test").glob("*_rec.png"))]
    rows = []
    for exp in exps:
        done = subprocess.run([sys.executable, __file__, "--child", exp, "--n", str(args.n)],
                              capture_output=True, text=True)
        if done.returncode:
            print(f"{exp}: failed\n{done.stderr[-2000:]}")
            continue
        result = json.loads(done.stdout.strip().splitlines()[-1])
        (experiments.experiment_dir(exp) / "benchmark.json").write_text(json.dumps(result, indent=1) + "\n")
        rows.append(result)
        print(f"{exp}: {result['seconds_median']:.3f} s/print on {result['device_name']}, "
              f"GPU {result['peak_gpu_mb'] or 0:.0f} MB, RAM {result['peak_ram_mb']:.0f} MB")
    if not rows:
        return
    # The table lists every experiment's benchmark.json, not only the ones measured now.
    files = [experiments.experiment_dir(e) / "benchmark.json" for e in experiments.list_experiments()]
    table = pd.DataFrame([json.loads(f.read_text()) for f in files if f.exists()])
    shown = pd.DataFrame({
        "experiment": table["experiment"], "device": table["device_name"],
        "s / print (median)": table["seconds_median"].map("{:.3f}".format),
        "s / print (p90)": table["seconds_p90"].map("{:.3f}".format),
        "peak GPU MB": table["peak_gpu_mb"].map(lambda v: "–" if pd.isna(v) else f"{v:.0f}"),
        "peak RAM MB": table["peak_ram_mb"].map("{:.0f}".format),
        "parameters": table["parameters"].map("{:,}".format)})
    out = config.EXPERIMENTS_DIR / "benchmark.md"
    out.write_text("# Repair cost\n\nWritten by `experiments/benchmark.py` (see its docstring for the protocol): "
                   f"{args.n} test prints per method, one at a time, after {WARMUP} warm-up calls; peak GPU = "
                   "PyTorch's peak allocated memory during the timed calls; peak RAM = the process's peak resident "
                   "memory, model and data included.\n\n" + data_md.table_to_markdown(shown) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
