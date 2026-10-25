"""Cross-experiment figures (matplotlib, PNG), from evaluate.py's run folders.

Used by evaluation/compare_experiments.py. Each experiment's own report is
written by evaluation/evaluate.py; these figures put several experiments side
by side. Damaged (floor) and clean (ceiling) come from the runs themselves.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

import config
from fpr import experiments
from fpr.data import eval_sets, image_io
from fpr.data.degradations import CONTAMINANTS
from fpr.metrics.matching import verification


def experiment_color(exp: str) -> str:
    """Fixed colour of an experiment, by its position among all experiment folders."""
    position = experiments.list_experiments().index(exp)
    return config.EXPERIMENT_COLORS[position % len(config.EXPERIMENT_COLORS)]


def style(ax: plt.Axes) -> None:
    """Recessive axes: light grid behind the data, no top/right spines, muted ticks."""
    ax.grid(color=config.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=config.MUTED_INK)


def save(fig: plt.Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=config.FIGURE_DPI)
    plt.close(fig)
    return out


def genuine_impostor(scores: pd.DataFrame, condition: str, contaminant: str | None = None):
    """(genuine, impostor) bozorth3 scores of one condition, optionally one contaminant."""
    rows = scores[scores["probe"] == condition]
    if contaminant is not None:
        if condition == "original":          # original probes carry no type: select by id
            ids = set(scores.loc[(scores["probe"] == "degraded") & (scores["type"] == contaminant), "id"])
            rows = rows[rows["id"].isin(ids)]
        else:
            rows = rows[rows["type"] == contaminant]
    return (rows.loc[rows["label"] == "genuine", "score"].to_numpy(),
            rows.loc[rows["label"] == "impostor", "score"].to_numpy())


def curves(runs: dict[str, Path]) -> list[tuple[str, str, Path, str]]:
    """(label, condition, run folder, colour) for damaged, each experiment, clean."""
    first = next(iter(runs.values()))
    return ([("damaged (floor)", "degraded", first, config.FLOOR_COLOR)]
            + [(exp, "reconstruction", folder, experiment_color(exp)) for exp, folder in runs.items()]
            + [("clean (ceiling)", "original", first, config.CEILING_COLOR)])


def run_eer(folder: Path, condition: str, contaminant: str = "all") -> float:
    """EER exactly as evaluate.py reports it (its matching.csv), so figures match the reports."""
    match = pd.read_csv(folder / "matching.csv")
    row = match[(match["type"] == contaminant) & (match["condition"] == condition)]
    return float(row["eer"].iloc[0])


def det_curves(runs: dict[str, Path], split: str, out: Path) -> Path:
    """DET (FRR vs FAR, log axes): floor, every experiment, ceiling."""
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    for label, condition, folder, color in curves(runs):
        scores = pd.read_csv(folder / "scores.csv", dtype={"id": str})
        points = verification.roc_points(*genuine_impostor(scores, condition))
        keep = (points["far"] > 0) & (points["frr"] > 0)
        ax.plot(points["far"][keep], points["frr"][keep], linewidth=2, color=color,
                label=f"{label} (EER {run_eer(folder, condition):.1%})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("false match rate (FAR)")
    ax.set_ylabel("false non-match rate (FRR)")
    ax.set_title(f"DET curves, {split} split", color=config.INK)
    style(ax)
    ax.legend(frameon=False)
    return save(fig, out)


def contaminant_eer_bars(runs: dict[str, Path], out: Path) -> Path:
    """Grouped bars: EER per contaminant for floor, each experiment and ceiling."""
    groups = curves(runs)
    width = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    x = np.arange(len(CONTAMINANTS))
    for k, (label, condition, folder, color) in enumerate(groups):
        eers = [run_eer(folder, condition, c) for c in CONTAMINANTS]
        ax.bar(x + (k - (len(groups) - 1) / 2) * width, np.array(eers) * 100, width * 0.9,
               color=color, label=label)
    ax.set_xticks(x, CONTAMINANTS)
    ax.set_ylabel("EER (%)")
    ax.set_title("Equal error rate by contaminant", color=config.INK)
    style(ax)
    ax.legend(frameon=False, ncols=min(len(groups), 4))
    return save(fig, out)


def qualitative_grid(split: str, exps: list[str], out: Path) -> Path:
    """One row per contaminant: clean / damaged / each experiment, mask outlined in red."""
    meta = eval_sets.load_meta(split)
    rows = [meta[meta["contaminant"] == c].iloc[0] for c in CONTAMINANTS]
    columns = ["clean", "damaged"] + exps
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(1.9 * len(columns), 2.6 * len(rows)))
    for r, row in enumerate(rows):
        mask = image_io.load_mask(eval_sets.triplet_path(split, row["sample_id"], "mask"))
        edge = mask & ~ndimage.binary_erosion(mask, iterations=2)
        paths = [eval_sets.triplet_path(split, row["sample_id"], "clean"),
                 eval_sets.triplet_path(split, row["sample_id"], "degraded")]
        paths += [experiments.reconstruction_path(e, split, row["probe_id"], row["contaminant"])
                  for e in exps]
        for c, (name, path) in enumerate(zip(columns, paths)):
            rgb = np.repeat(image_io.load_gray(path)[..., None], 3, axis=2)
            rgb[edge] = (227, 73, 72)
            axes[r, c].imshow(rgb)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(name, fontsize=10, color=config.INK)
        axes[r, 0].set_ylabel(row["contaminant"], fontsize=10, color=config.INK)
    return save(fig, out)
