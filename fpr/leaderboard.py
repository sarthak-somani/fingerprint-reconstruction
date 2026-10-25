"""One row per evaluation run, in run order: evaluation/results/leaderboard.csv + LEADERBOARD.md.

Called by evaluation/run_eval.py after evaluation/evaluate.py has written a run folder. Reads that
folder's headline.csv (scores 1-3, row 'all'), matching.csv (EER) and config.json (threshold,
benchmark). Each row records config.DATA_VERSION and EVAL_VERSION: rows that differ in either are
not comparable. Rows of the first metric set (restoration R, no normalisation) are archived in
leaderboard_eval1.csv / LEADERBOARD_eval1.md.
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from fpr.data import data_md

CSV = config.EVAL_RESULTS_DIR / "leaderboard.csv"
FIRST_DATA_VERSION = "v1"      # rows written before the data_version column existed
EVAL_VERSION = 2               # 2 = scores 1-3 on MSU-normalised images; 1 = archived R metrics


def run_row(exp: str, split: str, results_dir: Path) -> dict:
    """Headline numbers of one evaluate.py run (all contaminants together)."""
    head = pd.read_csv(results_dir / "headline.csv").set_index("type").loc["all"]
    match = pd.read_csv(results_dir / "matching.csv")
    match = match[match["type"] == "all"].set_index("condition")
    run_config = json.loads((results_dir / "config.json").read_text())
    bench = run_config.get("benchmark") or {}
    return {
        "run": f"{exp}/{results_dir.name}", "experiment": exp, "split": split,
        "data_version": config.DATA_VERSION, "eval_version": EVAL_VERSION,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tau": run_config["thresholds"][run_config["tau"]],
        "rescued": int(head["rescued"]), "broken": int(head["broken"]),
        "net_rescue": head["net_rescue"], "net_rescue_lo": head["net_rescue_lo"], "net_rescue_hi": head["net_rescue_hi"],
        "p_rescue": head["p_rescue"],
        "fmr_delta": head["fmr_delta"], "fmr_delta_lo": head["fmr_delta_lo"], "fmr_delta_hi": head["fmr_delta_hi"],
        "gate": head["gate"],
        "ridge_repaired": head["ridge_repaired"], "ridge_delta": head["ridge_delta"],
        "ridge_delta_lo": head["ridge_delta_lo"], "ridge_delta_hi": head["ridge_delta_hi"], "p_ridge": head["p_ridge"],
        "accept_clean": head["accept_clean"], "accept_damaged": head["accept_damaged"],
        "accept_repaired": head["accept_repaired"],
        "eer_clean": match.loc["original", "eer"], "eer_damaged": match.loc["degraded", "eer"],
        "eer_repaired": match.loc["reconstruction", "eer"],
        "seconds_per_print": bench.get("seconds_median"), "peak_gpu_mb": bench.get("peak_gpu_mb"),
        "peak_ram_mb": bench.get("peak_ram_mb"), "parameters": bench.get("parameters"), "device": bench.get("device"),
    }


def add_run(exp: str, split: str, results_dir: Path) -> None:
    """Append one run to leaderboard.csv and rewrite LEADERBOARD.md."""
    table = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame()
    table = pd.concat([table, pd.DataFrame([run_row(exp, split, results_dir)])], ignore_index=True)
    table.to_csv(CSV, index=False)
    write_markdown(table)


def interval(v: float, lo: float, hi: float, scale: float, digits: int) -> str:
    return f"{v * scale:+.{digits}f} [{lo * scale:+.{digits}f}, {hi * scale:+.{digits}f}]"


def write_markdown(table: pd.DataFrame) -> None:
    """LEADERBOARD.md: the same rows, the three scores with their CIs, then cost and EER."""
    shown = pd.DataFrame({
        "run": table["run"], "split": table["split"], "data": table["data_version"],
        "τ": table["tau"].map("{:g}".format),
        "rescued / broken": table["rescued"].astype(str) + " / " + table["broken"].astype(str),
        "1 net rescue (pts)": [interval(r.net_rescue, r.net_rescue_lo, r.net_rescue_hi, 100, 1) for r in table.itertuples()],
        "2 FMR change (pts)": [interval(r.fmr_delta, r.fmr_delta_lo, r.fmr_delta_hi, 100, 2) for r in table.itertuples()],
        "gate": table["gate"],
        "3 ridge fidelity change": [interval(r.ridge_delta, r.ridge_delta_lo, r.ridge_delta_hi, 1, 3)
                                    for r in table.itertuples()],
        "EER damaged → repaired (%)": [f"{100 * r.eer_damaged:.1f} → {100 * r.eer_repaired:.1f}" for r in table.itertuples()],
        "s / print": table["seconds_per_print"].map(lambda v: "–" if pd.isna(v) else f"{v:.2f}"),
        "GPU MB": table["peak_gpu_mb"].map(lambda v: "–" if pd.isna(v) else f"{v:.0f}"),
        "RAM MB": table["peak_ram_mb"].map(lambda v: "–" if pd.isna(v) else f"{v:.0f}"),
        "device": table["device"].fillna("–"),
    })
    text = ("# Leaderboard\n\n"
            "One row per evaluation run (`evaluation/run_eval.py`), oldest first. Each run's full report, "
            "including the definition of every metric (its section 0), is in `results/<run>/report.md`.\n\n"
            "- Every image is MSU-normalised before it is measured (temporary).\n"
            "- **τ**: the match-score threshold giving a 1% false-match rate on clean prints.\n"
            "- **1 net rescue**: (probes accepted only after repair − probes accepted only before) / probes, in "
            "percentage points, 95% CI.\n"
            "- **2 FMR change**: false-match rate after − before repair at τ, points, 95% CI; **gate** passes when "
            "the upper bound is below +0.5.\n"
            "- **3 ridge fidelity change**: local ridge correlation with the clean print in the damage mask, "
            "after − before repair, 95% CI.\n"
            "- Cost: median seconds per print, peak GPU / RAM, from `experiments/benchmark.py`.\n"
            "- `data`: the damage and val/test sets (`config.DATA_VERSION`). **Only compare rows with the same "
            "data version.** Older rows (restoration R, no normalisation) are in `LEADERBOARD_eval1.md`.\n\n"
            + data_md.table_to_markdown(shown) + "\n")
    config.LEADERBOARD_MD.write_text(text)
