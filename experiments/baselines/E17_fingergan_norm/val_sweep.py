"""Choose an output variant of a whole-print baseline on v3 val with the real evaluator (E17, E18, E19, E20).

sweep(exp, variants_fn, labels): for every val sample, variants_fn(degraded, mask) returns one repaired
image per label (e.g. whole print vs pasted back). Each label's repairs go to
experiments/<exp>/reconstructions/<version>/sweep_val/<label>/ (git-ignored), are scored by
evaluation/evaluate.py exactly as run_eval.py would (same layout, --rec-normalized when the outputs are
normalised prints), and the run folder is filed as evaluation/results/<exp>/val_sweep_<label>_<timestamp>/
(no leaderboard row: val runs are for selection only). run_eval() does run_eval.py's job for the test run.
Both use gallery_dir(), a readable stand-in for the v3 gallery copies (see there). Evaluations hold logs/.run_eval.lock, so
they never overlap another evaluation (run_eval.py picks up "the one new folder" in evaluation/results/).
Writes experiments/<exp>/sweep_val.csv and returns it; choose() picks the best label.
"""
import fcntl
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, when run as a script

import pandas as pd
from tqdm import tqdm

import config
from fpr import experiments
from fpr.data import eval_sets, image_io

REPO = Path(__file__).resolve().parents[3]
LOCK = REPO / "logs" / ".run_eval.lock"


TIMESTAMP = re.compile(r"^\d{8}_\d{6}$")              # evaluate.py's output folder name (as run_eval.py)


def timestamp_folders() -> set[str]:
    """Names of evaluate.py's not-yet-filed output folders in evaluation/results/."""
    return {p.name for p in config.EVAL_RESULTS_DIR.iterdir() if p.is_dir() and TIMESTAMP.match(p.name)}


def gallery_dir(split: str) -> Path:
    """The split's gallery layout folder, or a readable stand-in when its files cannot be read.

    preprocessing/data/v3/{val,test}/gallery/ holds byte-identical copies of the raw slap prints (script 07,
    shutil.copy2) whose permissions are owner/group only, so another account cannot read them. The stand-in,
    experiments/baselines/E17_fingergan_norm/cache/gallery/<version>/<split>/<probe_id>.png, links each name to the raw
    file itself (nist.raw_path): the same bytes, so the evaluator's content-addressed cache and every score
    are unchanged.
    """
    import os
    from fpr.data import nist
    layout = eval_sets.layout_dir(split, "gallery")
    if all(os.access(p, os.R_OK) for p in layout.glob("*.png")):
        return layout
    out = Path(__file__).resolve().parent / "cache" / "gallery" / config.DATA_VERSION / split
    out.mkdir(parents=True, exist_ok=True)
    meta = eval_sets.load_meta(split).drop_duplicates("probe_id")
    for row in meta.itertuples():
        link = out / eval_sets.layout_name(row.probe_id, row.contaminant, "gallery")
        if not link.is_symlink():
            link.symlink_to(nist.raw_path(row.gallery_path).resolve())
    assert len(list(out.glob("*.png"))) == len(meta), "stand-in gallery incomplete"
    return out


def evaluate(exp: str, split: str, recon_dir: Path, name: str, rec_normalized: bool,
             benchmark: Path | None = None) -> Path:
    """evaluate.py on one folder of repairs, under the evaluation lock; returns the filed run folder
    evaluation/results/<exp>/<split>_<name><timestamp>/."""
    command = [sys.executable, str(config.EVALUATE_SCRIPT),
               str(eval_sets.layout_dir(split, "original")), str(eval_sets.layout_dir(split, "degraded")),
               str(recon_dir), str(eval_sets.layout_dir(split, "mask")),
               "--gallery", str(gallery_dir(split))]
    if benchmark is not None and benchmark.exists():
        command += ["--benchmark", str(benchmark)]
    if rec_normalized:
        command.append("--rec-normalized")
    LOCK.parent.mkdir(exist_ok=True)
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = timestamp_folders()
        subprocess.run(command, check=True)
        new = sorted(timestamp_folders() - before)
        assert len(new) == 1, f"expected one new results folder, found {new}"
        filed = config.EVAL_RESULTS_DIR / exp / f"{split}_{name}{new[0]}"
        filed.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(config.EVAL_RESULTS_DIR / new[0]), filed)
    return filed


def evaluate_dir(exp: str, split: str, recon_dir: Path, label: str, rec_normalized: bool) -> dict:
    """One variant's val repairs -> the 'all' row of its headline.csv (plus run path); no leaderboard row."""
    filed = evaluate(exp, split, recon_dir, f"sweep_{label}_", rec_normalized)
    headline = pd.read_csv(filed / "headline.csv")
    row = headline[headline["type"] == "all"].iloc[0].to_dict()
    return {"label": label, **row, "run": str(filed.relative_to(REPO))}


def run_eval(exp: str, split: str = "test") -> Path:
    """evaluation/run_eval.py's job (benchmark, --rec-normalized from params.json, filing, leaderboard row),
    with gallery_dir() as the gallery. Usage: python experiments/baselines/E17_fingergan_norm/val_sweep.py EXX [split]"""
    from fpr import leaderboard
    experiments.check_data_version(exp)
    filed = evaluate(exp, split, experiments.reconstructions_dir(exp, split), "",
                     bool(experiments.load_params(exp).get("output_normalized")),
                     benchmark=experiments.experiment_dir(exp) / "benchmark.json")
    leaderboard.add_run(exp, split, filed)
    print(f"filed results in {filed}")
    return filed


def write_variants(exp: str, split: str, variants_fn: Callable, labels: list[str], limit: int | None = None
                   ) -> dict[str, Path]:
    """Repair every sample of `split` once and write each label's image to its own folder."""
    base = experiments.reconstructions_dir(exp, split).parent / f"sweep_{split}"
    dirs = {label: base / label for label in labels}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    meta = eval_sets.load_meta(split).head(limit)
    for row in tqdm(meta.itertuples(), total=len(meta), desc=f"{exp} {split} variants", mininterval=30):
        degraded = image_io.load_gray(eval_sets.triplet_path(split, row.sample_id, "degraded"))
        mask = image_io.load_mask(eval_sets.triplet_path(split, row.sample_id, "mask"))
        images = variants_fn(degraded, mask)
        name = eval_sets.layout_name(row.probe_id, row.contaminant, "reconstruction")
        for label in labels:
            image_io.save_png(dirs[label] / name, images[label])
    return dirs


def sweep(exp: str, variants_fn: Callable, labels: list[str], rec_normalized: dict[str, bool] | bool = True,
          split: str = "val") -> pd.DataFrame:
    """Write, evaluate and tabulate every variant; saves experiments/<exp>/sweep_<split>.csv."""
    dirs = write_variants(exp, split, variants_fn, labels)
    flags = rec_normalized if isinstance(rec_normalized, dict) else {label: rec_normalized for label in labels}
    rows = [evaluate_dir(exp, split, dirs[label], label, flags[label]) for label in labels]
    table = pd.DataFrame(rows)
    table.to_csv(experiments.experiment_dir(exp) / f"sweep_{split}.csv", index=False)
    print(table[["label", "rescued", "broken", "net_rescue", "fmr_delta", "gate", "ridge_repaired"]]
          .to_string(index=False), flush=True)
    return table


def choose(table: pd.DataFrame) -> str:
    """Highest net rescue (score 1) among variants that pass the gate; ties -> higher ridge fidelity."""
    ok = table[table["gate"] == "pass"]
    ok = ok if len(ok) else table
    return ok.sort_values(["net_rescue", "ridge_repaired"], ascending=False).iloc[0]["label"]


if __name__ == "__main__":
    run_eval(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "test")
