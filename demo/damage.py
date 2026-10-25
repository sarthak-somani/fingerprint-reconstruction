"""The demo's inputs: which print, how it is damaged, and which damage mask the model is given.

Prints come from the fixed test set (config.DEMO_SPLIT, data version config.DATA_VERSION) or an
upload. Damage is the stored test-set damage (exactly what the evaluator scored), one or several
synthetic contaminants stacked (fpr.data.degradations), or smudge strokes drawn in the editor. The mask
the model sees is the ground truth or the fine-tuned SegFormer's prediction (segmenter/).
"""
import json
from functools import lru_cache

import cv2
import numpy as np

import config
from fpr.data import eval_sets, foreground, image_io, nist
from fpr.data.degradations import apply_degradation

TEST_DAMAGE, CONTAMINANT_DAMAGE, DRAWN_DAMAGE = "test-set damage (as evaluated)", "contaminants", "draw a smudge"
DAMAGE_MODES = [TEST_DAMAGE, CONTAMINANT_DAMAGE, DRAWN_DAMAGE]
TRUE_MASK, PREDICTED_MASK = "ground truth", "predicted (fine-tuned SegFormer)"
MASK_SOURCES = [TRUE_MASK, PREDICTED_MASK]


@lru_cache(maxsize=1)
def samples():
    """meta.csv of the demo split indexed by sample_id (= probe_id in v2, '<probe>_<contaminant>' in v3)."""
    return eval_sets.load_meta(config.DEMO_SPLIT).set_index("sample_id")


def sample_ids() -> list[str]:
    return list(samples().index)


def load_sample(sample_id: str):
    """(clean uint8 (H, W), gallery print uint8 (H, W)) of one test sample."""
    row = samples().loc[sample_id]
    clean = image_io.load_gray(eval_sets.triplet_path(config.DEMO_SPLIT, sample_id, "clean"))
    try:                                   # the split's copy; the raw NIST file if it is missing or unreadable
        gallery = image_io.load_gray(eval_sets.gallery_copy_path(config.DEMO_SPLIT, row["gallery_path"]))
    except FileNotFoundError:
        gallery = image_io.load_gray(nist.raw_path(row["gallery_path"]))
    return clean, gallery


def to_gray(image: np.ndarray) -> np.ndarray:
    """Any numpy image from Gradio (grey, RGB or RGBA) -> uint8 (H, W)."""
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    return np.clip(image, 0, 255).astype(np.uint8)


def stacked_contaminants(clean: np.ndarray, names: list[str], severity: float, seed: int):
    """Apply each named contaminant in turn (one rng): (degraded uint8, union of their masks)."""
    fg = foreground.foreground_mask(clean)
    rng = np.random.default_rng(seed)
    degraded, mask = clean.copy(), np.zeros(clean.shape, bool)
    for name in names:
        degraded, damage = apply_degradation(degraded, fg, name, severity, rng)
        mask |= damage
    return degraded, mask


def drawn_smudge(editor: dict | None, clean: np.ndarray):
    """(degraded, mask) from the editor: the composite is the damaged print, painted pixels the mask."""
    layers = [np.asarray(layer) for layer in (editor or {}).get("layers") or [] if layer is not None]
    mask = np.zeros(clean.shape, bool)
    for layer in layers:
        painted = layer[..., 3] > 0 if layer.ndim == 3 and layer.shape[2] == 4 else to_gray(layer) > 0
        mask |= resize_to(painted.astype(np.uint8), clean.shape) > 0
    composite = editor.get("composite") if editor else None
    degraded = resize_to(to_gray(composite), clean.shape) if composite is not None else clean.copy()
    degraded[~mask] = clean[~mask]                       # only the strokes change the print
    return degraded, mask


def resize_to(image: np.ndarray, shape) -> np.ndarray:
    """Nearest-neighbour resize to (H, W) if the editor returned another canvas size."""
    if image.shape[:2] == tuple(shape):
        return image
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def damage(mode, sample_id, clean, names, severity, seed, editor):
    """(degraded uint8 (H, W), true damage mask bool (H, W), description) for the chosen damage mode."""
    if mode == TEST_DAMAGE:
        if sample_id is None:
            raise ValueError("Test-set damage needs a test print, not an upload: pick another damage mode.")
        row = samples().loc[sample_id]
        degraded = image_io.load_gray(eval_sets.triplet_path(config.DEMO_SPLIT, sample_id, "degraded"))
        mask = image_io.load_mask(eval_sets.triplet_path(config.DEMO_SPLIT, sample_id, "mask"))
        return degraded, mask, f"{row['contaminant']}, severity {row['severity']:.2f} (stored test sample)"
    if mode == CONTAMINANT_DAMAGE:
        if not names:
            raise ValueError("Tick at least one contaminant.")
        degraded, mask = stacked_contaminants(clean, names, severity, seed)
        return degraded, mask, f"{' + '.join(names)}, severity {severity:.2f}, seed {seed}"
    degraded, mask = drawn_smudge(editor, clean)
    if not mask.any():
        raise ValueError("Draw a smudge on the print in the editor first.")
    return degraded, mask, "hand-drawn smudge"


@lru_cache(maxsize=1)
def segmenter():
    """The fine-tuned SegFormer-B0 (segmenter/params.json: checkpoint, threshold), loaded once."""
    from segmenter.predict_masks import HERE, load_segmenter
    params = json.loads((HERE / "params.json").read_text())["model"]
    return load_segmenter(HERE / params["checkpoint"], params["threshold"])


def model_mask(source: str, degraded: np.ndarray, true_mask: np.ndarray) -> np.ndarray:
    """The mask the repair model is given: the true one, or the segmenter's prediction from the damaged print."""
    return true_mask if source == TRUE_MASK else segmenter().predict(degraded)

