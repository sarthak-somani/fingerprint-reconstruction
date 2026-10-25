"""Step 04: preview grid of every contaminant -> preprocessing/data/figures/degradation_preview.png.

Usage: python preprocessing/04_preview_degradations.py [--n-prints 6] [--seed 4] [--severity 1.0]
Rows: full L plain prints of train people at increasing severity (or all at --severity). Columns:
clean, then for each contaminant the damaged print and its mask (red overlay).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from fpr.data import foreground, image_io, nist, split
from fpr.data.degradations import apply_degradation, CONTAMINANTS


def mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGB float (H, W, 3) in [0, 1]: grey print with the mask tinted red."""
    rgb = np.repeat(image[..., None] / 255.0, 3, axis=2)
    rgb[mask] = 0.5 * rgb[mask] + 0.5 * np.array([1.0, 0.0, 0.0])
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-prints", type=int, default=6)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--severity", type=float, default=None,
                        help="one severity for every row (default: 0.2 -> 1.0 down the rows)")
    parser.add_argument("--out", type=Path, default=config.PREVIEW_FIGURE)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    index = nist.load_index()
    train = index[index["participant_id"].isin(split.participants_in("train"))
                  & (index["device"] == config.PROBE_DEVICE)]
    rows = train.iloc[rng.choice(len(train), args.n_prints, replace=False)]
    severities = (np.full(args.n_prints, args.severity) if args.severity is not None
                  else np.linspace(0.2, 1.0, args.n_prints))

    n_cols = 1 + 2 * len(CONTAMINANTS)
    fig, axes = plt.subplots(args.n_prints, n_cols, figsize=(1.6 * n_cols, 2.4 * args.n_prints))
    for r, (row, severity) in enumerate(zip(rows.itertuples(), severities)):
        clean = image_io.load_gray(nist.raw_path(row.path))
        fg = foreground.foreground_mask(clean)
        axes[r, 0].imshow(clean, cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_ylabel(f"sev {severity:.2f}")
        for c, name in enumerate(CONTAMINANTS):
            degraded, mask = apply_degradation(clean, fg, name, severity, rng)
            axes[r, 1 + 2 * c].imshow(degraded, cmap="gray", vmin=0, vmax=255)
            axes[r, 2 + 2 * c].imshow(mask_overlay(degraded, mask))
            if r == 0:
                axes[r, 1 + 2 * c].set_title(name, fontsize=9)
                axes[r, 2 + 2 * c].set_title(f"{name} mask", fontsize=9)
    axes[0, 0].set_title("clean", fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
