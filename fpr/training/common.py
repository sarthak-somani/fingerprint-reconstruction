"""Pieces both trainers share: the learning-rate schedule, the fixed validation batch, the CSV logs,
the model / EMA / optimiser set-up and the command line.

Optimiser: AdamW (lr 2e-4, no weight decay), linear warm-up over 1,000 steps then cosine decay to 0,
gradient clipping at 1.0, EMA of the weights (decay 0.999), fp16 autocast on GPU; all from the
experiment's params.json 'training' block.
"""
import argparse
import copy
import csv
import math
from pathlib import Path

import numpy as np
import torch

import config
from fpr import experiments
from fpr.data import datasets, eval_sets, foreground, image_io, patches
from fpr.models.unet import UNet

SMOKE = dict(steps=20, batch_size=4, workers=0, log_every=5, checkpoint_every=10, val_patches=8, warmup_steps=5)


def lr_factor(step: int, train: dict) -> float:
    """Linear warm-up, then cosine decay to 0 at the last step."""
    if step < train["warmup_steps"]:
        return (step + 1) / train["warmup_steps"]
    progress = (step - train["warmup_steps"]) / max(1, train["steps"] - train["warmup_steps"])
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def val_batch(n: int, seed: int, normalized: bool = False) -> dict:
    """n fixed 128 px patches from val samples (raw, or script 10's normalised copies), centred on the
    damage (same every run). The val people never appear in training (fpr/data/split.py)."""
    meta, rng, items = eval_sets.load_meta("val"), np.random.default_rng(seed), []
    path = eval_sets.normalized_triplet_path if normalized else eval_sets.triplet_path
    for k in range(n):
        sample = meta.iloc[k % len(meta)]["sample_id"]
        clean, degraded = (image_io.load_gray(path("val", sample, kind)) for kind in ("clean", "degraded"))
        mask = image_io.load_gray(eval_sets.triplet_path("val", sample, "mask"))
        mask, size = mask > 127, config.PATCH_SIZE
        top, left = patches.patch_corner(mask, rng, size, min_fg=0.0)
        window = (slice(top, top + size), slice(left, left + size))
        fg = foreground.foreground_mask(degraded)[window]
        c, d, m = clean[window], degraded[window], mask[window]
        items.append({"clean": datasets.to_tensor(c), "degraded": datasets.to_tensor(d),
                      "mask": datasets.to_tensor(m),
                      "orientation": torch.from_numpy(datasets.filled_orientation(d, m, fg))})
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def build_model(params: dict, train: dict, device: torch.device):
    """(model, ema, optimiser, scaler), made in this order so every trainer's random streams match E02's."""
    model = UNet(**params["model"]).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimiser = torch.optim.AdamW(model.parameters(), lr=train["lr"], weight_decay=train["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=train["amp"] and device.type == "cuda")
    return model, ema, optimiser, scaler


def resume(out: Path, model, ema, optimiser, scaler, device) -> int:
    """Load out/last.pt if it exists (a preempted job resubmitted); returns the step to continue from."""
    if not (out / "last.pt").exists():
        return 0
    state = torch.load(out / "last.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"]), ema.load_state_dict(state["ema"])
    optimiser.load_state_dict(state["optimiser"]), scaler.load_state_dict(state["scaler"])
    print(f"resumed at step {state['step']}")
    return state["step"]


def optimiser_step(loss, model, ema, optimiser, scaler, train: dict) -> None:
    """Backward (fp16-scaled), clip, AdamW step, EMA update."""
    optimiser.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimiser)
    torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip"])
    scaler.step(optimiser)
    scaler.update()
    with torch.no_grad():
        for p_ema, p in zip(ema.parameters(), model.parameters()):
            p_ema.lerp_(p, 1 - train["ema_decay"])


def parse_args(doc: str, extra=None) -> tuple[argparse.Namespace, dict, dict, Path]:
    """--exp NAME [--out DIR] [--smoke] [--steps N] -> (args, params, training block, output folder).

    --smoke: 20 tiny CPU steps into <exp>/checkpoints_smoke/, to test the pipeline end to end.
    """
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--exp", required=True, help="experiment name, e.g. E02_resshift")
    parser.add_argument("--out", type=Path, default=None, help="default experiments/<group>/<exp>/checkpoints/")
    parser.add_argument("--smoke", action="store_true", help="20 tiny steps on CPU, to test the pipeline")
    parser.add_argument("--steps", type=int, default=None, help="override params.json (e.g. a short GPU check)")
    if extra:
        extra(parser)
    args = parser.parse_args()
    params = experiments.load_params(args.exp)
    train = dict(params["training"])
    if args.steps:
        train.update(steps=args.steps, checkpoint_every=args.steps)
    out = args.out or experiments.experiment_dir(args.exp) / "checkpoints"
    if args.smoke:
        train.update(SMOKE)
        out = out.parent / "checkpoints_smoke"
    return args, params, train, out
