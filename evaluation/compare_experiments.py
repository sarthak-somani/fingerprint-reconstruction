"""Put experiments side by side -> evaluation/results/comparison/{split}/ (comparison/<version>/{split}/ off v2).

Usage: python evaluation/compare_experiments.py [--split test] [--exp E02_resshift E04_...]
Uses each experiment's latest evaluate.py run for that split, data version and eval version (run
evaluation/run_eval.py first). Writes a DET plot, per-contaminant EER bars, a qualitative grid,
results_table.md / .tex (the leaderboard row per experiment) and comparison.md: every experiment's
headline scores, and every pair of experiments (and the damaged probes) compared on score 1 and
score 3 with significance tests (see pairwise()).

With --log-table EXX it only prints EXX's per-contaminant table for experiments/experimental_log.md,
next to the main baseline E04_msu_stft (descriptive, no tests).
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable
sys.path.insert(0, str(Path(__file__).resolve().parent))      # evaluate.py: shared statistics

import numpy as np
import pandas as pd

import config
import evaluate
from fpr import experiments, figures, leaderboard
from fpr.data import data_md

BASELINE = "E04_msu_stft"      # the main baseline: the experimental log describes every run against it
DAMAGED = "damaged"            # the unrepaired probes, compared like an experiment
EQUIV_NET = 0.03               # score 1: |difference| < 3 points (about 12 of 398 probes) counts as equivalent
EQUIV_RIDGE = 0.01             # score 3: |difference| < 0.01 counts as equivalent
ALPHA = 0.05


def latest_runs(split: str, exps: list[str]) -> dict[str, Path]:
    """{experiment: newest results folder for this split, config.DATA_VERSION and the current
    leaderboard.EVAL_VERSION}, skipping experiments without one."""
    table = pd.read_csv(leaderboard.CSV)
    current = set(table.loc[(table["data_version"] == config.DATA_VERSION)
                            & (table["eval_version"] == leaderboard.EVAL_VERSION), "run"])
    runs = {}
    for exp in exps:
        folders = sorted((config.EVAL_RESULTS_DIR / exp).glob(f"{split}_*"))
        folders = [f for f in folders if (f / "probes.csv").exists() and f"{exp}/{f.name}" in current]
        if folders:
            runs[exp] = folders[-1]
        else:
            print(f"skip {exp}: no {split} results for data {config.DATA_VERSION}, eval {leaderboard.EVAL_VERSION}")
    return runs


def write_tables(runs: dict[str, Path], out_dir: Path) -> None:
    """results_table.md / .tex: the leaderboard row of each experiment's latest run."""
    table = pd.read_csv(leaderboard.CSV)
    wanted = {f"{exp}/{folder.name}" for exp, folder in runs.items()}
    table = table[table["run"].isin(wanted)].drop(columns=["run", "date"]).round(4)
    (out_dir / "results_table.md").write_text(data_md.table_to_markdown(table) + "\n")
    (out_dir / "results_table.tex").write_text(table.to_latex(index=False, float_format="%.4f"))


def pts(v: float, digits: int = 1) -> str:
    return f"{100 * v:+.{digits}f}"


def summary_table(runs: dict[str, Path]) -> str:
    """One row per experiment: the three scores (all probes), EER and cost, from its run folder."""
    rows = []
    for exp, run in runs.items():
        h = pd.read_csv(run / "headline.csv").set_index("type").loc["all"]
        cfg = json.loads((run / "config.json").read_text())
        e, b = cfg["eer_change"], cfg.get("benchmark") or {}
        gpu = b.get("peak_gpu_mb")
        rows.append([f"**{exp}**" if exp == BASELINE else exp, int(h["rescued"]), int(h["broken"]),
                     f"{pts(h['net_rescue'])} [{pts(h['net_rescue_lo'])}, {pts(h['net_rescue_hi'])}]",
                     evaluate.pvalue(h["p_rescue"]),
                     f"{pts(h['fmr_delta'], 2)} [{pts(h['fmr_delta_lo'], 2)}, {pts(h['fmr_delta_hi'], 2)}]", h["gate"],
                     f"{h['ridge_repaired']:.3f}",
                     f"{h['ridge_delta']:+.3f} [{h['ridge_delta_lo']:+.3f}, {h['ridge_delta_hi']:+.3f}]",
                     f"{100 * e['eer_repaired']:.1f}",
                     "–" if not b else f"{b['seconds_median']:.2f} {b['device']}",
                     "–" if gpu is None else f"{gpu:.0f}", "–" if not b else f"{b['peak_ram_mb']:.0f}"])
    first = next(iter(runs.values()))
    h = pd.read_csv(first / "headline.csv").set_index("type").loc["all"]
    e = json.loads((first / "config.json").read_text())["eer_change"]
    header = ["experiment", "rescued", "broken", "1 net rescue, pts (95% CI)", "p", "2 FMR change, pts (95% CI)",
              "gate", "3 ridge fidelity", "3 change (95% CI)", "EER %", "s / print", "GPU MB", "RAM MB"]
    fingers = f", {int(h['n_fingers'])} fingers" if "n_fingers" in h and h["n_fingers"] != h["n"] else ""
    note = (f"\n\nAll samples (n = {int(h['n'])}{fingers}). Reference points: accepted at τ: clean {h['accept_clean']:.1%}, "
            f"damaged {h['accept_damaged']:.1%}; FMR damaged {h['fmr_damaged']:.2%}; ridge fidelity damaged "
            f"{h['ridge_damaged']:.3f} (clean = 1); EER damaged {100 * e['eer_damaged']:.1f}%. p = exact sign-flip test over fingers "
            "(rescued vs broken; = exact McNemar with one sample per finger). CIs resample fingers.")
    return md_table(header, rows) + note


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def probe_values(runs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(accepted at τ, ridge fidelity in the mask) per sample (rows) and experiment (columns), plus
    the damaged samples as column DAMAGED, and each sample's finger id (the statistical unit).
    Every run must use the same τ and samples."""
    accepted, ridge, taus = {}, {}, set()
    for exp, run in runs.items():
        p = pd.read_csv(run / "probes.csv", dtype={"id": str}).set_index("key").sort_index()
        ids = p["id"]
        taus.add(json.loads((run / "config.json").read_text())["thresholds"][evaluate.TAU])
        accepted[exp], ridge[exp] = p["acc_rec"], p["ridge_region_rec"]
        accepted[DAMAGED], ridge[DAMAGED] = p["acc_deg"], p["ridge_region_deg"]
    assert len(taus) == 1, f"runs use different thresholds {taus}: re-evaluate them"
    accepted, ridge = pd.DataFrame(accepted), pd.DataFrame(ridge)
    assert accepted.notna().all().all(), "runs cover different probes"
    return accepted, ridge, ids.loc[accepted.index]


def pairwise(runs: dict[str, Path]) -> pd.DataFrame:
    """Every pair (A, B) of experiments and the damaged probes, on score 1 and score 3.

    Score 1: Δ = acceptance rate of A − of B at τ (the net rescue rates differ by exactly this, since
    both start from the same damaged probes); exact sign-flip test on the per-finger sums of the accepted
    differences (= exact McNemar with one sample per finger). Score 3: Δ = mean ridge fidelity of A − of B;
    Wilcoxon signed-rank test of the per-finger mean differences. CIs resample fingers with all their
    samples (evaluate.boot_means). p-values are Holm-adjusted
    over all pairs, separately per score. Verdict: 'equivalent' when the 90% bootstrap CI of Δ lies within
    ±EQUIV_* (two one-sided tests at 5%), even if the difference is significant; else 'A better' /
    'B better' when the adjusted p < ALPHA; otherwise 'unresolved'.
    """
    accepted, ridge, ids = probe_values(runs)
    ids = ids.to_numpy()
    names = [DAMAGED] + list(runs)
    rows = []
    for a, b in itertools.combinations(names, 2):
        d1 = accepted[a].to_numpy(float) - accepted[b].to_numpy(float)
        d3 = (ridge[a] - ridge[b]).to_numpy()
        row = {"A": a, "B": b}
        for score, d, p in (("net", d1, evaluate.sign_flip_p(d1, ids)),
                            ("ridge", d3, evaluate.wilcoxon_p(d3, ids))):
            means = evaluate.boot_means(d, ids, "pairwise")
            row |= {f"{score}_delta": d.mean(), f"{score}_lo": np.percentile(means, 2.5),
                    f"{score}_hi": np.percentile(means, 97.5), f"{score}_lo90": np.percentile(means, 5),
                    f"{score}_hi90": np.percentile(means, 95), f"{score}_p": p}
        rows.append(row)
    table = pd.DataFrame(rows)
    for score, margin in (("net", EQUIV_NET), ("ridge", EQUIV_RIDGE)):
        table[f"{score}_p_holm"] = holm(table[f"{score}_p"].to_numpy())
        # equivalence first: a difference inside the margin is negligible even when n makes it significant
        table[f"{score}_verdict"] = [
            "equivalent" if -margin < r[f"{score}_lo90"] and r[f"{score}_hi90"] < margin
            else ("A better" if r[f"{score}_delta"] > 0 else "B better") if r[f"{score}_p_holm"] < ALPHA
            else "unresolved" for _, r in table.iterrows()]
    return table


SYMBOL = {"A better": "▲", "B better": "▼", "equivalent": "≈", "unresolved": "·"}


def verdict_matrix(table: pd.DataFrame, score: str, names: list[str]) -> str:
    """Row vs column: ▲ row better, ▼ row worse, ≈ equivalent, · unresolved."""
    cell = {}
    for _, r in table.iterrows():
        v = r[f"{score}_verdict"]
        cell[(r["A"], r["B"])] = SYMBOL[v]
        cell[(r["B"], r["A"])] = SYMBOL[{"A better": "B better", "B better": "A better"}.get(v, v)]
    short = [n.split("_")[0] if n != DAMAGED else "dmg" for n in names]
    rows = [[short[i]] + ["" if a == b else cell[(a, b)] for b in names] for i, a in enumerate(names)]
    return md_table([""] + short, rows)


def pair_table(table: pd.DataFrame) -> str:
    rows = []
    for _, r in table.iterrows():
        rows.append([r["A"], r["B"],
                     f"{pts(r['net_delta'])} [{pts(r['net_lo'])}, {pts(r['net_hi'])}]",
                     evaluate.pvalue(r["net_p_holm"]), r["net_verdict"],
                     f"{r['ridge_delta']:+.3f} [{r['ridge_lo']:+.3f}, {r['ridge_hi']:+.3f}]",
                     evaluate.pvalue(r["ridge_p_holm"]), r["ridge_verdict"]])
    return md_table(["A", "B", "1: A − B, pts (95% CI)", "p (Holm)", "verdict", "3: A − B (95% CI)", "p (Holm)",
                     "verdict"], rows)


def comparison_markdown(runs: dict[str, Path], split: str) -> str:
    table = pairwise(runs)
    names = [DAMAGED] + list(runs)
    sources = ", ".join(f"`{r.relative_to(config.REPO_ROOT)}`" for r in runs.values())
    return "\n".join([
        f"### All experiments ({split})", "", summary_table(runs), "",
        "### Every pair", "",
        f"Score 1 compares acceptance at τ (exact sign-flip test on per-finger sums; = McNemar with one sample per "
        f"finger), score 3 the ridge fidelity in the mask (Wilcoxon signed-rank on per-finger means), sample by "
        f"sample; CIs resample fingers. p-values are Holm-adjusted over all {len(table)} pairs, per score. "
        f"**equivalent**: the 90% CI of the difference lies within ±{100 * EQUIV_NET:g} points (score 1) or "
        f"±{EQUIV_RIDGE:g} (score 3), whatever the p-value. **A better / B better**: otherwise, adjusted "
        f"p < {ALPHA:g}. **unresolved**: neither. "
        f"`{DAMAGED}` = the unrepaired probes. Score 2 is a per-run gate (table above).", "",
        "**Score 1 (identity recovery)**: row vs column, ▲ row better, ▼ row worse, ≈ equivalent, · unresolved", "",
        verdict_matrix(table, "net", names), "",
        "**Score 3 (ridge fidelity)**: same reading", "",
        verdict_matrix(table, "ridge", names), "",
        "<details><summary>All pairs with differences, CIs and adjusted p-values</summary>", "",
        pair_table(table), "", "</details>", "",
        f"Source: {sources}.", ""])


def log_table(exp: str, split: str) -> str:
    """Per-contaminant scores of EXX for the experimental log, 'EXX / E04' per cell (EXX alone for E04)."""
    exps = [exp] if exp == BASELINE else [exp, BASELINE]
    runs = latest_runs(split, exps)
    assert list(runs) == exps, f"need {split} results for {exps}"
    heads = [pd.read_csv(runs[e] / "headline.csv").set_index("type") for e in exps]
    columns = [("rescued", lambda r: f"{int(r['rescued'])}"), ("broken", lambda r: f"{int(r['broken'])}"),
               ("1 net rescue (pts)", lambda r: pts(r["net_rescue"])),
               ("2 FMR change (pts)", lambda r: pts(r["fmr_delta"], 2)),
               ("3 ridge fidelity", lambda r: f"{r['ridge_repaired']:.3f}"),
               ("3 change", lambda r: f"{r['ridge_delta']:+.3f}")]
    rows = []
    for t in heads[0].index:
        h = heads[0].loc[t]
        rows.append([f"**{t}**" if t == "all" else t, str(int(h["n"])),
                     f"{h['accept_clean']:.0%} / {h['accept_damaged']:.0%}", f"{h['ridge_damaged']:.3f}"]
                    + [" / ".join(fn(hd.loc[t]) for hd in heads) for _, fn in columns])
    header = ["contaminant", "n", "accepted clean / damaged", "ridge fid. damaged"] + [c for c, _ in columns]
    runs_text = ", ".join(f"`{runs[e].relative_to(config.REPO_ROOT)}`" for e in exps)
    return md_table(header, rows) + f"\n\nCells: {' / '.join(exps)}. Source: {runs_text}.\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=config.EVAL_SPLITS)
    parser.add_argument("--exp", nargs="*", default=None, help="default: every experiment")
    parser.add_argument("--log-table", metavar="EXX", help="only print EXX's table for experiments/experimental_log.md")
    args = parser.parse_args()
    if args.log_table:
        print(log_table(args.log_table, args.split))
        return

    runs = latest_runs(args.split, args.exp or experiments.list_experiments())
    if not runs:
        return
    out_dir = config.COMPARISON_DIR / (args.split if config.DATA_VERSION == "v2" else f"{config.DATA_VERSION}/{args.split}")
    for name, path in runs.items():
        print(f"using {name}: {path.relative_to(config.REPO_ROOT)}")
    print("wrote", figures.det_curves(runs, args.split, out_dir / "det.png"))
    print("wrote", figures.contaminant_eer_bars(runs, out_dir / "eer_by_contaminant.png"))
    print("wrote", figures.qualitative_grid(args.split, list(runs), out_dir / "qualitative.png"))
    write_tables(runs, out_dir)
    print(f"wrote {out_dir / 'results_table.md'} / .tex")
    (out_dir / "comparison.md").write_text(comparison_markdown(runs, args.split))
    print(f"wrote {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
