"""Evaluate one E27 run with evaluation/evaluate.py, exactly as run_eval.py would, outside evaluation/results/.

Usage:  FPR_DATA_VERSION=v3 python experiments/main/E27_seed_study/evaluate_run.py --run E22_s1_r2 [--workers N] [--root DIR]
Same evaluator (imported unchanged, its main()), same arguments as run_eval.run_evaluator: v3 test layout
(original, degraded, the run's repairs, mask), --gallery, --benchmark = the model's experiment's benchmark.json
(cost only), no --rec-normalized (E09 / E22 repair raw prints; the evaluator normalises everything). Only its
output folder and its cache folder are redirected, so nothing is added to evaluation/results/ or LEADERBOARD.md
and several runs can be evaluated at once:
- output: eval/<run>/ (report.md, probes.csv, headline.csv, ...); the rebuildable intermediate folders
  minutiae/ and normalized/ (the repairs' MINDTCT files and normalised copies) are removed afterwards
  unless --keep-intermediates;
- cache: a private copy of evaluation/results/degraded_cache/'s tables (taken under the evaluation lock
  logs/.run_eval.lock), with its content-addressed minutiae/ and normalized/ folders linked read-mostly
  (v3 test's originals, gallery and damaged prints are all in there already). If the evaluator's settings
  differ from the shared cache's (e.g. NFIQ 2 missing on this node), the cache is private and empty instead:
  slower, same numbers.
"""
import argparse
import fcntl
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: makes config/fpr importable

import config
from experiments.main.E27_seed_study import runs
from fpr.data import eval_sets

EVAL_LOCK = config.REPO_ROOT / "logs" / ".run_eval.lock"
SHARED_CACHE = config.EVALUATION_DIR / "results" / "degraded_cache"
CACHE_TABLES = ("degraded.csv", "scores.csv", "nfiq2.json", "meta.json")
CACHE_FOLDERS = ("minutiae", "normalized")
INTERMEDIATES = ("minutiae", "normalized")


def load_evaluator():
    """evaluation/evaluate.py as a module (it is a script, not a package member)."""
    spec = importlib.util.spec_from_file_location("evaluate", config.EVALUATION_DIR / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate"] = module          # so its process pools can pickle its functions (normalize_image)
    spec.loader.exec_module(module)
    return module


def evaluator_args(name: str, root: Path = runs.HERE) -> list[str]:
    """run_eval.run_evaluator's command line for this run's repairs."""
    model = runs.parse(name)[0]
    args = [str(eval_sets.layout_dir(runs.SPLIT, "original")), str(eval_sets.layout_dir(runs.SPLIT, "degraded")),
            str(runs.recon_dir(name, root)), str(eval_sets.layout_dir(runs.SPLIT, "mask")),
            "--gallery", str(eval_sets.layout_dir(runs.SPLIT, "gallery"))]
    if runs.benchmark_json(model).exists():
        args += ["--benchmark", str(runs.benchmark_json(model))]
    return args


def private_cache(evaluate, work: Path) -> Path:
    """A cache folder for this run: copies of the shared tables + links to its content-addressed folders."""
    cache = work / "degraded_cache"
    cache.mkdir(parents=True)
    tools = {name: evaluate.find_tool(name) for name in ("mindtct", "bozorth3", "nfiq2")}
    wanted = evaluate.text_sha1(json.dumps(evaluate.settings_dict(tools), sort_keys=True))
    EVAL_LOCK.parent.mkdir(exist_ok=True)
    with open(EVAL_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            shared = json.loads((SHARED_CACHE / "meta.json").read_text()).get("settings_hash")
            if shared != wanted:
                print(f"evaluator settings {wanted[:10]} != shared cache {str(shared)[:10]} (NFIQ 2 {tools['nfiq2']}): "
                      "private empty cache, everything is recomputed", flush=True)
                return cache
            for name in CACHE_TABLES:
                shutil.copy2(SHARED_CACHE / name, cache / name)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    for name in CACHE_FOLDERS:
        (cache / name).symlink_to(SHARED_CACHE / name, target_is_directory=True)
    return cache


def evaluate_run(name: str, root: Path = runs.HERE, workers: int | None = None, keep: bool = False) -> Path:
    assert config.DATA_VERSION == runs.DATA_VERSION, "run with FPR_DATA_VERSION=v3"
    assert runs.repaired(name, root), f"{runs.recon_dir(name, root)} is incomplete; run sample.py first"
    out = runs.eval_dir(name, root)
    work = out.parent / f".work_{name}"
    shutil.rmtree(work, ignore_errors=True)             # a crashed earlier attempt (never the shared cache: links only)
    evaluate = load_evaluator()
    evaluate.RESULTS_DIR = work / "results"
    evaluate.CACHE_DIR = private_cache(evaluate, work)
    evaluate.EVAL_DIR = work      # only used after import to print the cache path relative to it (report.md)
    if workers:
        evaluate.N_WORKERS = workers
    sys.argv = ["evaluate.py"] + evaluator_args(name, root)
    print(f"{name}: evaluate.py {' '.join(sys.argv[1:])}", flush=True)
    evaluate.main()
    produced = [p for p in evaluate.RESULTS_DIR.iterdir() if p.is_dir()]
    assert len(produced) == 1, produced
    if out.exists():
        shutil.rmtree(out)
    shutil.move(str(produced[0]), out)
    if not keep:
        for sub in INTERMEDIATES:
            shutil.rmtree(out / sub, ignore_errors=True)
    for name_ in CACHE_FOLDERS:                         # unlink, never follow, the shared folders
        link = evaluate.CACHE_DIR / name_
        if link.is_symlink():
            link.unlink()
    shutil.rmtree(work)
    (out / "e27_run.json").write_text(json.dumps({"run": name, "evaluator_args": evaluator_args(name, root),
                                                  "recon_dir": str(runs.recon_dir(name, root))}, indent=1) + "\n")
    print(f"{name}: results in {out}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--root", type=Path, default=runs.HERE)
    parser.add_argument("--workers", type=int, default=None, help="evaluate.py's thread / process pool size")
    parser.add_argument("--keep-intermediates", action="store_true")
    args = parser.parse_args()
    evaluate_run(args.run, args.root, args.workers or int(os.environ.get("E27_EVAL_WORKERS", 0)) or None,
                 args.keep_intermediates)


if __name__ == "__main__":
    main()
