"""What the demo's buttons do. Plain functions of the widget values, so tests call them without a browser.

run_demo: damage the print -> choose the mask (true or segmented) -> repair with one or two models ->
score every version the way the evaluator does (demo/scoring.py) -> pictures + a metrics table.
"""
import random
import time
from functools import lru_cache

import numpy as np
import torch

from demo import damage, models, scoring, views

NO_MODEL = "(none)"


def pick_print(sample_id: str | None, upload: np.ndarray | None):
    """(clean uint8, gallery uint8 or None, sample_id or None): an upload wins over the dropdown."""
    if upload is not None:
        return damage.to_gray(upload), None, None
    clean, gallery = damage.load_sample(sample_id)
    return clean, gallery, sample_id


def editor_value(sample_id: str | None, upload: np.ndarray | None) -> dict:
    """A fresh drawing canvas showing the chosen clean print."""
    clean = pick_print(sample_id, upload)[0]
    return {"background": clean, "layers": [], "composite": clean}


def random_sample() -> str:
    """A random test print for the 'random print' button."""
    return random.choice(damage.sample_ids())


@lru_cache(maxsize=32)
def gallery_view(sample_id: str) -> np.ndarray:
    """The normalised gallery print of a test sample (normalised once per sample)."""
    return scoring.evaluator_view(damage.load_sample(sample_id)[1])


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def score_row(label, image_n, clean_n, true_mask, matcher, key, extra="", measure_region=True):
    """One markdown table row: match score and verdict, ridge fidelity and PSNR in the damage, time."""
    score = matcher.score(image_n, key) if matcher else None
    match = "n/a (no NBIS)" if score is None else f"{score:.0f} → **{scoring.verdict(score)}**"
    ridge = f"{scoring.ridge_fidelity(clean_n, image_n, true_mask):.3f}" if measure_region else "1 (reference)"
    psnr = f"{scoring.psnr(clean_n, image_n, true_mask):.1f}" if measure_region else "—"
    return f"| {label} | {match} | {ridge} | {psnr} | {extra} |"


def repair_with(name, degraded, used_mask, clean_n, true_mask, matcher):
    """(repaired print to show, |diff| heat map, table row) for one model."""
    model, load_seconds = models.load_model(name)
    repaired, seconds = models.run_model(model, degraded, used_mask)
    repaired_n = repaired if models.output_normalized(name) else scoring.evaluator_view(repaired)
    timing = (f"{seconds:.2f} s on {models.device_of(model)}"
              + (f" (+{load_seconds:.1f} s loading, first use only)" if load_seconds >= 0.1 else "")
              + f"; benchmark {models.benchmark_line(name)}")
    label = f"**{name}**" + (" (output already normalised)" if models.output_normalized(name) else "")
    row = score_row(label, repaired_n, clean_n, true_mask, matcher, name, timing)
    return repaired, views.difference(repaired_n, clean_n, true_mask), row


def warm_up(names: list[str], sample_id: str) -> None:
    """Run every model once before serving, so no live click pays for loading and CUDA start-up (~30 s).

    On CPU only the fast models are warmed: a CPU run of E02 alone takes minutes.
    """
    if not torch.cuda.is_available():
        names = [name for name in names if name in ("E04_msu_stft", "E09_ccdf", "E22_ccdf_aux_minutiae_cldice")]
    for name in names:
        start = time.perf_counter()
        run_demo(sample_id, None, None, damage.TEST_DAMAGE, [], 0.6, 0, damage.PREDICTED_MASK, name, NO_MODEL)
        print(f"warmed up {name} in {time.perf_counter() - start:.1f} s", flush=True)


def run_demo(sample_id, upload, editor, mode, contaminants, severity, seed, mask_source, model_a, model_b):
    """Everything the 'Damage and repair' button shows: 7 pictures and a markdown report.

    Raises ValueError with a readable message when the inputs do not make sense.
    """
    clean, gallery, sample_id = pick_print(sample_id, upload)
    degraded, true_mask, how = damage.damage(mode, sample_id, clean, list(contaminants or []),
                                             float(severity), int(seed), editor)
    start = time.perf_counter()
    used_mask = damage.model_mask(mask_source, degraded, true_mask)
    mask_seconds = time.perf_counter() - start
    predicted = mask_source == damage.PREDICTED_MASK

    clean_n = scoring.evaluator_view(clean)
    reference = "gallery (slap) print" if gallery is not None else "your clean upload"
    matcher = None
    if scoring.nbis_available():
        matcher = scoring.Matcher(gallery_view(sample_id) if gallery is not None else clean_n)
    tau = "not evaluated yet" if scoring.tau() is None else f"{scoring.tau():g}"
    rows = [f"| version | bozorth3 vs {reference} (τ = {tau}) | ridge fidelity in damage ↑ | "
            f"PSNR in damage, dB ↑ | time |", "|---|---|---|---|---|"]
    if gallery is not None:
        rows.append(score_row("clean print (ceiling)", clean_n, clean_n, true_mask, matcher, "clean",
                              measure_region=False))
    rows.append(score_row("damaged, no repair (floor)", scoring.evaluator_view(degraded), clean_n, true_mask,
                          matcher, "damaged"))

    pictures = []
    for name in (model_a, model_b):
        if name and name != NO_MODEL:
            repaired, heat, row = repair_with(name, degraded, used_mask, clean_n, true_mask, matcher)
            pictures += [repaired, heat]
            rows.append(row)
        else:
            pictures += [None, None]

    notes = [f"**Damage:** {how}; {true_mask.mean():.0%} of the image."]
    if predicted:
        notes.append(f"**Mask:** SegFormer prediction in {mask_seconds:.2f} s, Dice {dice(used_mask, true_mask):.2f} "
                     "against the true mask (cyan outline = predicted, red = true).")
    if "E04_msu_stft" in (model_a, model_b):
        notes.append("E04 ignores the mask: it enhances the whole print (MSU's pipeline).")
    notes.append("All scores are on MSU-normalised prints, as in evaluation/evaluate.py; MATCH = score > τ "
                 "(FMR 1% on clean test prints). Ridge fidelity and PSNR are measured in the true damage mask.")
    report = "\n".join(rows) + "\n\n" + "\n\n".join(notes)
    return [clean, views.outlined(degraded, true_mask, used_mask if predicted else None),
            views.mask_picture(used_mask, true_mask), *pictures, report]
