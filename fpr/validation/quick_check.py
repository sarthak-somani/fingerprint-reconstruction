"""quick_check(experiment, split): fast validation EER for picking checkpoints during training.

The main evaluation is evaluation/evaluate.py (run via evaluation/run_eval.py).
This is the lighter, importable check the training loop can call. It compares
each probe against every slap print of its split (not a random 50), so its EER
differs from evaluate.py's. Results go to evaluation/results/<exp>/quick_check_<split>/.
"""
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from fpr import experiments
from fpr.data import eval_sets, image_io, nist
from fpr.metrics import image_metrics, minutiae
from fpr.metrics.matching import nbis, verification


def probe_metrics(exp: str, split: str, row, xyt_dir, with_images: bool) -> dict:
    """Minutiae rates (and optionally image metrics) for one sample; caches its .xyt files."""
    clean = image_io.load_gray(eval_sets.triplet_path(split, row.sample_id, "clean"))
    mask = image_io.load_mask(eval_sets.triplet_path(split, row.sample_id, "mask"))
    output = image_io.load_gray(experiments.reconstruction_path(exp, split, row.probe_id, row.contaminant))

    _, recon_min = nbis.run_mindtct(output, xyt_dir / row.sample_id)
    clean_root = config.XYT_CACHE_DIR / "clean" / split / row.probe_id
    if not clean_root.with_suffix(".min").exists():
        nbis.run_mindtct(clean, clean_root)
    rates = minutiae.compare_minutiae(minutiae.read_min_file(recon_min),
                                      minutiae.read_min_file(clean_root.with_suffix(".min")), mask)
    images = image_metrics.all_image_metrics(output, clean, mask) if with_images else {}
    return {"sample_id": row.sample_id, "probe_id": row.probe_id, "contaminant": row.contaminant,
            **images, **rates}


def pair_scores(split: str, xyt_dir, meta: pd.DataFrame) -> pd.DataFrame:
    """bozorth3 score of every sample against every gallery print its probe is paired with."""
    pairs = pd.read_csv(eval_sets.pairs_path(split), dtype={"probe_id": str})
    gallery_dir = config.XYT_CACHE_DIR / "gallery"
    gallery_xyt = {path: nbis.cached_xyt(nist.raw_path(path), gallery_dir)
                   for path in pairs["gallery_path"].unique()}
    groups, scores = dict(tuple(pairs.groupby("probe_id"))), []
    for row in tqdm(meta.itertuples(), total=len(meta), desc="bozorth3", mininterval=10):
        group = groups[row.probe_id]
        files = [gallery_xyt[path] for path in group["gallery_path"]]
        scores.append(group.assign(sample_id=row.sample_id, contaminant=row.contaminant,
                                   score=nbis.bozorth_scores(xyt_dir / f"{row.sample_id}.xyt", files)))
    return pd.concat(scores, ignore_index=True)


def split_scores(scores: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(genuine, impostor) score arrays from a scores table."""
    return (scores.loc[scores["label"] == 1, "score"].to_numpy(),
            scores.loc[scores["label"] == 0, "score"].to_numpy())


def matching_summary(scores: pd.DataFrame) -> dict:
    """EER and TAR at each FAR in config.FAR_TARGETS."""
    genuine, impostor = split_scores(scores)
    summary = {"eer": verification.eer(genuine, impostor)}
    for far in config.FAR_TARGETS:
        summary[f"tar_at_far_{far:g}"] = verification.tar_at_far(genuine, impostor, far)
    return summary


def save_match_threshold(exp: str, scores: pd.DataFrame) -> None:
    """Store the bozorth3 threshold at val FAR = DEMO_FAR for `exp` (the demo's MATCH line)."""
    path = config.MATCH_THRESHOLD_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = json.loads(path.read_text()) if path.exists() else {}
    thresholds[exp] = verification.threshold_at_far(*split_scores(scores), config.DEMO_FAR)
    path.write_text(json.dumps(thresholds, indent=2))


def quick_check(exp: str, split: str = "val", with_images: bool = False) -> dict:
    """Validation EER of experiments/<group>/<exp>/reconstructions/{split}/. Returns a summary dict.

    with_images=True also computes PSNR/SSIM/LPIPS/orientation error (slower).
    Writes per_probe.csv, scores.csv, summary.json and, for val, match_threshold.json.
    """
    nbis.require_nbis()
    out_dir = config.EVAL_RESULTS_DIR / exp / f"quick_check_{split}"
    xyt_dir = config.XYT_CACHE_DIR / "runs" / f"{exp}_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = eval_sets.load_meta(split)

    per_probe = pd.DataFrame([probe_metrics(exp, split, row, xyt_dir, with_images)
                              for row in tqdm(meta.itertuples(), total=len(meta), desc="minutiae",
                                              mininterval=10)])
    per_probe.to_csv(out_dir / "per_probe.csv", index=False)
    scores = pair_scores(split, xyt_dir, meta)
    scores.to_csv(out_dir / "scores.csv", index=False)

    by_contaminant = {name: matching_summary(group)["eer"] for name, group in scores.groupby("contaminant")}
    numeric = per_probe.drop(columns=["sample_id", "probe_id", "contaminant", "n_clean", "n_recon"])
    summary = {"experiment": exp, "split": split, **matching_summary(scores),
               **numeric.mean().round(4).to_dict(), "eer_by_contaminant": by_contaminant}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if split == "val":
        save_match_threshold(exp, scores)
    return summary
