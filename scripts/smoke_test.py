"""End-to-end CPU smoke test without data: damage a synthetic print, repair it with a final model, score it.

Usage:  python scripts/smoke_test.py [--exp E22_ccdf_aux_minutiae_cldice] [--contaminant mehendi]
Builds the experiment's reconstructor (its trained checkpoint if present, else a randomly initialised
network, which tests the plumbing, not the quality), damages a synthetic 300 x 240 ridge pattern with one
contaminant (fpr/data/degradations), repairs it (tiling -> CCDF sampling -> paste-back) and reports the
evaluator's ridge fidelity and PSNR inside the damage mask, on MSU-normalised images as evaluate.py does.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import numpy as np

from demo import scoring
from fpr import experiments
from fpr.data import foreground
from fpr.data.degradations import CONTAMINANTS, apply_degradation


def synthetic_print(h: int = 300, w: int = 240) -> np.ndarray:
    """uint8 (h, w): curved ridges (period ~9 px, as at 500 ppi) inside an elliptical fingertip on white."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ridges = 0.5 + 0.5 * np.cos(2 * np.pi * np.hypot(xx - w / 2, yy - 1.3 * h) / 9.0)
    inside = ((xx - w / 2) / (0.42 * w)) ** 2 + ((yy - h / 2) / (0.45 * h)) ** 2 < 1
    return np.where(inside, 40 + 180 * (1 - ridges), 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="E22_ccdf_aux_minutiae_cldice")
    parser.add_argument("--contaminant", default="mehendi", choices=CONTAMINANTS)
    args = parser.parse_args()

    clean = synthetic_print()
    degraded, mask = apply_degradation(clean, foreground.foreground_mask(clean), args.contaminant, 0.6,
                                       np.random.default_rng(0))
    try:
        method, how = experiments.build_reconstructor(args.exp), "trained checkpoint"
    except FileNotFoundError:
        method, how = experiments.build_reconstructor(args.exp, random_init=True), "random weights (no checkpoint)"
    start = time.perf_counter()
    repaired = method.repair(degraded, mask)
    seconds = time.perf_counter() - start
    assert repaired.shape == clean.shape and repaired.dtype == np.uint8
    assert np.array_equal(repaired[~mask], degraded[~mask]), "pixels outside the mask changed"

    clean_n, damaged_n, repaired_n = (scoring.evaluator_view(x) for x in (clean, degraded, repaired))
    print(f"{args.exp} ({type(method).__name__}, {how}, {method.num_parameters():,} parameters, "
          f"{getattr(method, 'device', 'cpu')}): {args.contaminant}, {mask.mean():.1%} of the print damaged, "
          f"repaired in {seconds:.1f} s")
    for label, image in (("damaged", damaged_n), ("repaired", repaired_n)):
        print(f"  {label:<9} ridge fidelity {scoring.ridge_fidelity(clean_n, image, mask):+.3f}   "
              f"PSNR {scoring.psnr(clean_n, image, mask):.2f} dB   (inside the mask)")
    print("ok: pixels outside the mask are unchanged")


if __name__ == "__main__":
    main()
