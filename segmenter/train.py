"""Fine-tune SegFormer-B0 (ImageNet MiT-b0 encoder, new 1-class head) to predict the damage mask.

Usage (GPU):  python segmenter/train.py
Quick local check:                 python segmenter/train.py --smoke
Data: FixedTrainDataset (the stored damaged train set, train people only), `patch` px windows, 90% on
damage. It refuses to run on the v1 train set (no 'lotion' samples). Loss: BCE + soft Dice. AdamW,
head lr 10x encoder lr (as SegFormer), linear warm-up then poly decay. Every `val_every` steps:
pooled Dice on the full v2 val probes; best.pt keeps the best. Writes segmenter/checkpoints/.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr importable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fpr.data import datasets, image_io
from segmenter import data, metrics
from segmenter.predict import DamageSegmenter, normalise
from segmenter.segformer import load_released

HERE = Path(__file__).resolve().parent
PARAMS = json.loads((HERE / "params.json").read_text())


def loss_fn(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """BCE + soft Dice; logits (B, 1, H/4, W/4) are upsampled to the mask's (B, 1, H, W)."""
    logits = F.interpolate(logits, size=mask.shape[2:], mode="bilinear", align_corners=False)
    probability = torch.sigmoid(logits)
    dice = 1 - (2 * (probability * mask).sum((1, 2, 3)) + 1) / (probability.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logits, mask) + dice.mean()


def lr_factor(step: int, train: dict) -> float:
    """Linear warm-up, then poly decay (power 1) to 0 at the last step."""
    if step < train["warmup_steps"]:
        return (step + 1) / train["warmup_steps"]
    return max(0.0, 1 - (step - train["warmup_steps"]) / max(1, train["steps"] - train["warmup_steps"]))


def val_dice(model, meta, threshold: float) -> float:
    """Pooled Dice of the predicted masks on full v2 val probes."""
    segmenter, total = DamageSegmenter(model, threshold), {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for row in meta.itertuples():
        degraded = image_io.load_gray(data.probe_path("v2", "val", row.probe_id, "degraded"))
        truth = image_io.load_mask(data.probe_path("v2", "val", row.probe_id, "mask"))
        for name, value in metrics.confusion(segmenter.predict(degraded), truth).items():
            total[name] += value
    model.train()
    return metrics.scores(total)["dice"]


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="10 tiny steps on CPU, to test the pipeline")
    args = parser.parse_args()
    train = dict(PARAMS["training"])
    out = HERE / ("checkpoints_smoke" if args.smoke else "checkpoints")
    val_meta = data.load_meta("v2", "val")
    if args.smoke:
        train.update(steps=10, batch_size=2, workers=0, log_every=5, val_every=10, warmup_steps=2)
        val_meta = val_meta.head(2)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(train["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = datasets.FixedTrainDataset(patch=train["patch"], seed=train["seed"])
    assert args.smoke or "lotion" in set(dataset.meta["contaminant"]), \
        "train/damaged/ is still the v1 set (no lotion): wait for script 08's v2 run"
    loader = DataLoader(dataset, batch_size=train["batch_size"], shuffle=True, drop_last=True,
                        num_workers=train["workers"], pin_memory=device.type == "cuda",
                        persistent_workers=train["workers"] > 0)
    model = load_released(PARAMS["model"]["pretrained_encoder"], PARAMS["model"]["num_labels"]).to(device).train()
    head = [p for n, p in model.named_parameters() if n.startswith("decode_head.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("decode_head.")]
    optimiser = torch.optim.AdamW([{"params": body, "lr": train["lr"]}, {"params": head, "lr": train["lr"] * 10}],
                                  weight_decay=train["weight_decay"])
    base_lrs = [group["lr"] for group in optimiser.param_groups]
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, {len(dataset)} train samples, device {device}")

    step, best = 0, -1.0
    while step < train["steps"]:
        dataset.set_epoch(step // len(loader))
        for batch in loader:
            for group, lr in zip(optimiser.param_groups, base_lrs):
                group["lr"] = lr * lr_factor(step, train)
            mask = batch["mask"].to(device)
            loss = loss_fn(model(normalise(batch["degraded"].to(device))), mask)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            step += 1
            if step % train["log_every"] == 0:
                append_csv(out / "log.csv", {"step": step, "loss": round(loss.item(), 5)})
                print(f"step {step}  loss {loss.item():.4f}", flush=True)
            if step % train["val_every"] == 0 or step == train["steps"]:
                dice = val_dice(model, val_meta, PARAMS["model"]["threshold"])
                append_csv(out / "val.csv", {"step": step, "val_dice": round(dice, 4)})
                state = {"model": model.state_dict(), "step": step, "val_dice": dice, "params": PARAMS}
                torch.save(state, out / "last.pt")
                if dice > best:
                    best = dice
                    torch.save(state, out / "best.pt")
                print(f"step {step}  val pooled Dice {dice:.4f}  (best {best:.4f})", flush=True)
            if step >= train["steps"]:
                break


if __name__ == "__main__":
    main()
