"""E27 analysis: per-run scores, seed spread, hierarchical bootstrap CIs and the figure, from eval/<run>/probes.csv.

Usage:  python experiments/main/E27_seed_study/analysis.py [--root DIR]     (runs at the end of every chain.sh post stage)
Score 1 = net rescue (accepted repaired - accepted damaged at tau), score 2 = FMR change gate, score 3 = ridge
fidelity in the damage mask: evaluation/evaluate.py's own functions (headline_scores, cluster_sums, boot_index).
Hierarchical bootstrap (N_BOOT = 10,000): per resample, draw training seeds with replacement, sampler seeds
with replacement within each drawn training seed (independently per model), and fingers with replacement with
all their samples (one finger draw shared by every model in that resample: paired by finger, as
evaluate.py's v3 statistics). With one run per model it is exactly evaluate.py's finger bootstrap. E04 (deterministic) and the
damaged probes get only the finger draw. Writes results.md, results.csv (per run), bootstrap.csv, seed_study.png.
Partial grids are fine: whatever is evaluated is used, and results.md says how many runs it pools.
"""
import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: makes config/fpr importable

import numpy as np
import pandas as pd

import config
from experiments.main.E27_seed_study import runs

REFERENCE = {"E04": "E04_msu_stft", "E09": "E09_ccdf", "E22": "E22_ccdf_aux_minutiae_cldice"}
SCORES = {"s1": "score 1 net rescue (pts)", "s3": "score 3 ridge fidelity"}
COMPARISONS = [("E09", "E04"), ("E22", "E04"), ("E09", "damaged"), ("E22", "damaged"), ("E22", "E09")]
SEED_COLORS = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}     # evaluate.py's validated categorical slots 1-3
SEED_MARKERS = {0: "o", 1: "s", 2: "D"}


def load_evaluator():
    spec = importlib.util.spec_from_file_location("evaluate", config.EVALUATION_DIR / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate"] = module          # so its process pools can pickle its functions (normalize_image)
    spec.loader.exec_module(module)
    return module


EV = load_evaluator()


def read_probes(folder: Path) -> pd.DataFrame:
    return pd.read_csv(folder / "probes.csv", dtype={"id": str}).sort_values("key").reset_index(drop=True)


def reference_run(exp: str, version: str = runs.DATA_VERSION) -> Path | None:
    """The newest evaluation/results/<exp>/ run of v3 test under the current eval version (leaderboard.csv)."""
    from fpr import leaderboard
    table = pd.read_csv(leaderboard.CSV)
    rows = table[(table["experiment"] == exp) & (table["split"] == runs.SPLIT) & (table["data_version"] == version)
                 & (table["eval_version"] == leaderboard.EVAL_VERSION)]
    return config.EVAL_RESULTS_DIR / sorted(rows["run"])[-1] if len(rows) else None


def sample_scores(p: pd.DataFrame) -> dict:
    """Per-sample contributions: s1 = acc_rec - acc_deg (mean = net rescue), s3 = ridge_region_rec."""
    return {"s1": p["acc_rec"].to_numpy(float) - p["acc_deg"].to_numpy(float), "s3": p["ridge_region_rec"].to_numpy(float)}


def finger_counts(n_fingers: int, seed_text: str) -> np.ndarray:
    """(N_BOOT, F) how often each finger is drawn, from evaluate.boot_index (same draws as its boot_means)."""
    index = EV.boot_index(n_fingers, seed_text)
    counts = np.zeros((index.shape[0], n_fingers))
    np.add.at(counts, (np.repeat(np.arange(index.shape[0]), n_fingers), index.ravel()), 1)
    return counts


def run_boot(values: list[np.ndarray], ids: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """(N_BOOT, K) finger-bootstrap means of each run's per-sample values, all with the same finger draws."""
    sums = np.stack([EV.cluster_sums(v, ids)[0] for v in values], axis=1)       # (F, K)
    n = EV.cluster_sums(np.ones(len(ids)), ids)[1]                               # samples per finger
    return (draws @ sums) / (draws @ n)[:, None]


def seed_draws(grid: dict[int, list[int]], n_boot: int, label: str) -> np.ndarray:
    """(N_BOOT, S*R) column indices into the model's runs (ordered by training seed, then sampler seed):
    S training seeds drawn with replacement, then R sampler seeds with replacement within each (ragged ok)."""
    seeds = sorted(grid)
    n_r = np.array([len(grid[s]) for s in seeds])
    offsets = np.concatenate([[0], np.cumsum(n_r)[:-1]])
    width = int(n_r.max())
    rng = np.random.default_rng([EV.SEED, int(hashlib.sha1(label.encode()).hexdigest()[:8], 16)])
    ts = rng.integers(0, len(seeds), size=(n_boot, len(seeds)))
    if len(seeds) == 1 and width == 1:
        return np.zeros((n_boot, 1), int)
    pick = np.floor(rng.random((n_boot, len(seeds), width)) * n_r[ts][:, :, None]).astype(int)
    return (offsets[ts][:, :, None] + pick).reshape(n_boot, -1)


def hierarchical(model_values: dict[str, dict[tuple[int, int], np.ndarray]], ids: np.ndarray, seed_text: str
                 ) -> dict[str, np.ndarray]:
    """{model: (N_BOOT,) bootstrap means}: seeds resampled per model, fingers shared across models."""
    draws = finger_counts(len(pd.unique(ids)), seed_text)
    out = {}
    for model, cells in model_values.items():
        keys = sorted(cells)
        boot = run_boot([cells[k] for k in keys], ids, draws)
        grid = {}
        for s, r in keys:
            grid.setdefault(s, []).append(r)
        cols = seed_draws(grid, len(draws), f"{seed_text}/{model}")
        out[model] = np.take_along_axis(boot, cols, axis=1).mean(axis=1)
    return out


def per_run_row(name: str, p: pd.DataFrame) -> dict:
    h = EV.headline_scores(p).set_index("type").loc["all"]
    model, s, r = runs.parse(name) if runs.NAME.match(name) else (name, -1, -1)
    return {"run": name, "model": model, "train_seed": s, "sampler_seed": r, "rescued": int(h["rescued"]),
            "broken": int(h["broken"]), "net_rescue": h["net_rescue"], "net_rescue_lo": h["net_rescue_lo"],
            "net_rescue_hi": h["net_rescue_hi"], "fmr_delta": h["fmr_delta"], "fmr_delta_hi": h["fmr_delta_hi"],
            "gate": h["gate"], "ridge_fidelity": h["ridge_repaired"], "ridge_delta": h["ridge_delta"],
            "ridge_delta_lo": h["ridge_delta_lo"], "ridge_delta_hi": h["ridge_delta_hi"]}


def spread(table: pd.DataFrame, col: str) -> dict:
    """Mean, SD across training seeds (of their sampler-seed means), mean SD across sampler seeds, min, max."""
    by_seed = table.groupby("train_seed")[col]
    within = by_seed.std(ddof=1).dropna()
    return {"runs": len(table), "train_seeds": table["train_seed"].nunique(), "mean": table[col].mean(),
            "sd_train": by_seed.mean().std(ddof=1) if by_seed.ngroups > 1 else np.nan,
            "sd_sampler": within.mean() if len(within) else np.nan, "min": table[col].min(), "max": table[col].max()}


def analyse(root: Path = runs.HERE, reference_dirs: dict | None = None) -> dict:
    names = [n for n in runs.all_runs() if runs.evaluated(n, root)]
    assert names, f"nothing evaluated under {root / 'eval'}"
    probes = {n: read_probes(runs.eval_dir(n, root)) for n in names}
    refs = reference_dirs if reference_dirs is not None else {k: reference_run(v) for k, v in REFERENCE.items()}
    ref_probes = {k: read_probes(v) for k, v in refs.items() if v is not None}
    base = probes[names[0]]
    for label, p in list(probes.items()) + list(ref_probes.items()):
        assert p["key"].equals(base["key"]) and p["id"].equals(base["id"]), f"{label}: different samples"
        assert p["acc_deg"].equals(base["acc_deg"]), f"{label}: different tau or damaged scores"
        assert np.allclose(p["ridge_region_deg"], base["ridge_region_deg"]), f"{label}: different damaged ridge"
        assert not p[["acc_rec", "ridge_region_rec"]].isna().any().any(), f"{label}: NaN scores"
    ids = base["id"].to_numpy()

    table = pd.DataFrame([per_run_row(n, p) for n, p in probes.items()])
    ref_rows = pd.DataFrame([per_run_row(f"{k} (evaluation/results)", p) for k, p in ref_probes.items()])
    damaged = {"s1": 0.0, "s3": base["ridge_region_deg"].mean()}
    e04 = {k: v.mean() for k, v in sample_scores(ref_probes["E04"]).items()} if "E04" in ref_probes else None

    boots, point = {}, {}
    for score in SCORES:
        values = {}
        for n, p in probes.items():
            model, s, r = runs.parse(n)
            values.setdefault(model, {})[(s, r)] = sample_scores(p)[score]
        values["damaged"] = {(0, 0): np.zeros(len(ids)) if score == "s1" else base["ridge_region_deg"].to_numpy(float)}
        if e04 is not None:
            values["E04"] = {(0, 0): sample_scores(ref_probes["E04"])[score]}
        boots[score] = hierarchical(values, ids, f"E27/{score}")
        point[score] = {m: float(np.mean([v.mean() for v in cells.values()])) for m, cells in values.items()}

    rows = []
    for score in SCORES:
        b = boots[score]
        for model in [m for m in runs.MODELS if m in b] + [m for m in ("E04", "damaged") if m in b]:
            rows.append({"score": score, "what": model, "estimate": point[score][model],
                         "lo": np.percentile(b[model], 2.5), "hi": np.percentile(b[model], 97.5), "p_le_0": np.nan})
        for a, c in COMPARISONS:
            if a in b and c in b:
                d = b[a] - b[c]
                rows.append({"score": score, "what": f"{a} - {c}", "estimate": point[score][a] - point[score][c],
                             "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5), "p_le_0": float((d <= 0).mean())})
    boot_table = pd.DataFrame(rows)

    col = {"s1": "net_rescue", "s3": "ridge_fidelity"}
    spreads = pd.DataFrame([{"model": m, "score": s, **spread(table[table["model"] == m], col[s])}
                            for m in runs.MODELS if (table["model"] == m).any() for s in SCORES])
    pairs = []
    if {"E09", "E22"} <= set(table["model"]):
        for s in SCORES:
            seed_means = {m: table[table["model"] == m].groupby("train_seed")[col[s]].mean() for m in ("E09", "E22")}
            a, b_ = seed_means["E22"], seed_means["E09"]
            cross = [(x > y) for x in a for y in b_]
            matched = [a[k] > b_[k] for k in a.index.intersection(b_.index)]
            run_a, run_b = (table[table["model"] == m][col[s]].to_numpy() for m in ("E22", "E09"))
            pairs.append({"score": s, "train_seed_pairs": f"{sum(cross)}/{len(cross)}",
                          "matched_seed_pairs": f"{sum(matched)}/{len(matched)}",
                          "run_pairs": f"{int((run_a[:, None] > run_b[None, :]).sum())}/{run_a.size * run_b.size}"})
    beats = []
    if e04 is not None:
        for m in runs.MODELS:
            t = table[table["model"] == m]
            for s in SCORES:
                if len(t):
                    target = e04[s] if s == "s3" else e04["s1"]
                    seed_means = t.groupby("train_seed")[col[s]].mean()
                    beats.append({"model": m, "score": s, "runs_above_E04": f"{int((t[col[s]] > target).sum())}/{len(t)}",
                                  "train_seeds_above_E04": f"{int((seed_means > target).sum())}/{len(seed_means)}",
                                  "gate_pass": f"{int((t['gate'] == 'pass').sum())}/{len(t)}"})
    sanity = []
    for m in runs.MODELS:
        n = runs.run_name(m, 0, 0)
        if n in probes and m in ref_probes:
            a, o = probes[n], ref_probes[m]
            sanity.append({"run": n, "reference": str(refs[m].relative_to(config.REPO_ROOT)),
                           "same_acceptance": float((a["acc_rec"] == o["acc_rec"]).mean()),
                           "max_ridge_diff": float((a["ridge_region_rec"] - o["ridge_region_rec"]).abs().max())})
    return {"table": table, "ref_rows": ref_rows, "boot": boot_table, "spreads": spreads, "pairs": pd.DataFrame(pairs),
            "beats": pd.DataFrame(beats), "sanity": pd.DataFrame(sanity), "damaged": damaged, "e04": e04,
            "n_fingers": len(pd.unique(ids)), "n_samples": len(ids), "boots": boots}


def md(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "(none yet)"
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)])


def figure(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table, boot = res["table"], res["boot"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    for ax, score, col, scale in ((axes[0], "s1", "net_rescue", 100), (axes[1], "s3", "ridge_fidelity", 1)):
        models = [m for m in runs.MODELS if (table["model"] == m).any()]
        for i, m in enumerate(models):
            t = table[table["model"] == m]
            for s, g in t.groupby("train_seed"):
                x = i - 0.2 + 0.12 * s + 0.02 * (g["sampler_seed"].to_numpy() - 1)
                ax.scatter(x, g[col] * scale, s=36, color=SEED_COLORS.get(s, "#52514e"), marker=SEED_MARKERS.get(s, "o"),
                           edgecolor="white", linewidth=0.8, zorder=3, label=f"training seed {s}" if i == 0 else None)
            row = boot[(boot["score"] == score) & (boot["what"] == m)].iloc[0]
            ax.errorbar(i + 0.25, row["estimate"] * scale, yerr=[[(row["estimate"] - row["lo"]) * scale],
                        [(row["hi"] - row["estimate"]) * scale]], fmt="o", color="#0b0b0b", ms=6, capsize=4, lw=2,
                        zorder=4, label="pooled mean, 95% CI" if i == 0 else None)
        refs = [("damaged", res["damaged"][score], ":")]
        if res["e04"] is not None:
            refs.append(("E04 (main baseline)", res["e04"][score], "--"))
        for label, value, style in refs:
            ax.axhline(value * scale, color="#52514e", ls=style, lw=1.2, zorder=1)
            ax.text(len(models) - 0.5, value * scale, f" {label}", va="bottom", ha="right", color="#52514e", fontsize=9)
        ax.set_xticks(range(len(models)), models)
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.set_ylabel("net rescue (points)" if score == "s1" else "ridge fidelity in the damage mask")
        ax.set_title(SCORES[score], fontsize=11)
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"E27: per-run scores on v3 test (training seeds x sampler seeds); hierarchical bootstrap 95% CI",
                 fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write(res: dict, root: Path) -> None:
    res["table"].to_csv(root / "results.csv", index=False)
    res["boot"].to_csv(root / "bootstrap.csv", index=False)
    figure(res, root / "seed_study.png")
    t = res["table"]
    lines = ["# E27 seed study: results (written by analysis.py; do not edit by hand)", "",
             f"v3 test, {res['n_samples']} samples / {res['n_fingers']} fingers. {len(t)} of "
             f"{len(runs.all_runs())} runs evaluated: " + ", ".join(
                 f"{m}: training seeds {sorted(t[t['model'] == m]['train_seed'].unique().tolist())}, {int((t['model'] == m).sum())} runs"
                 for m in runs.MODELS if (t["model"] == m).any()) + ".",
             "Sampler configuration fixed (selected once on val for training seed 0: one-call x0 start, t0 = 2, projection on);"
             " no re-selection per seed. Score 1 in fractions (x100 = points).", "",
             "## Per run (evaluate.py headline_scores, finger bootstrap within the run)", "", md(t), "",
             "Reference runs (evaluation/results, v3 test):", "", md(res["ref_rows"]), "",
             "## Spread across seeds", "", "sd_train = SD of the training seeds' means (over sampler seeds); "
             "sd_sampler = mean within-training-seed SD over sampler seeds.", "", md(res["spreads"]), "",
             "## Hierarchical bootstrap (training seeds > sampler seeds > fingers, 10,000 resamples)", "",
             "estimate = mean over all runs; lo / hi = 95% percentile CI; p_le_0 = share of resamples with a difference <= 0. "
             "Differences are paired by finger (one finger draw per resample for every model); each model's seeds are drawn "
             "independently. With 3 training seeds the seed level is resampled from 3 values only, so read the CI as a "
             "sensitivity check, not an exact interval.", "", md(res["boot"]), "",
             "## E22 vs E09 across seeds", "", "train_seed_pairs: all (E22 seed, E09 seed) pairs of training-seed means; "
             "matched: same training seed (E02 and E07 share init and data order at a given seed); run_pairs: every "
             "(E22 run, E09 run).", "", md(res["pairs"]), "",
             "## Every seed against E04", "", md(res["beats"]), "",
             "## Sanity: seed-0 / sampler-seed-0 runs vs the original E09 / E22 v3 evaluations", "",
             "Same checkpoint, configuration and seed; differences can only come from GPU nondeterminism.", "",
             md(res["sanity"]), "", "![seeds](seed_study.png)", ""]
    (root / "results.md").write_text("\n".join(lines))
    print(f"wrote {root / 'results.md'}, results.csv, bootstrap.csv, seed_study.png ({len(t)} runs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=runs.HERE)
    args = parser.parse_args()
    write(analyse(args.root), args.root)


if __name__ == "__main__":
    main()
