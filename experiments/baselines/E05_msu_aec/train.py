"""Train E05_msu_aec with MSU-LatentAFIS's own recipe on our clean train prints.

Usage (GPU):  python experiments/baselines/E05_msu_aec/train.py
Quick local check:                 python experiments/baselines/E05_msu_aec/train.py --smoke
MSU's recipe (enhancement_AEC.py: ImageFromFile_AutoEcoder, Model, _get_optimizer): the label is a
128 px crop of a good print's STFT-enhanced texture image; the input is the same crop with random
brightness and contrast, Gaussian noise (sigma 1 on the [-1, 1] scale) and a random blur. L2 loss,
Adam lr 1e-4, betas (0.5, 0.9), batch 128. It never sees our contaminants. Writes checkpoints/ and cache/.
"""
import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: makes config/fpr importable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

import config
from experiments.baselines.E04_msu_stft.reconstructor import msu_stft
from experiments.baselines.E05_msu_aec.model import EnhancementAEC
from fpr.data import eval_sets, image_io

HERE = Path(__file__).resolve().parent
PARAMS = json.loads((HERE / "params.json").read_text())


def stft_cached(source: Path, target: Path) -> Path:
    """MSU STFT image of one clean print, saved once as uint8 PNG (padded to more than one patch)."""
    if not target.exists():
        image = image_io.pad_to_at_least(image_io.load_gray(source), PARAMS["training"]["patch"] + 1)
        cv2.imwrite(str(target), np.round(msu_stft(image, PARAMS["texture_sigma"], PARAMS["block"])).astype(np.uint8))
    return target


def build_cache(sources: list[Path], cache_dir: Path, workers: int) -> list[np.ndarray]:
    """STFT images of all `sources` (computed in parallel the first time), loaded into memory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    targets = [cache_dir / f"{Path(source).stem}.png" for source in sources]   # names are unique per print
    with Pool(workers) as pool:
        pool.starmap(stft_cached, zip(sources, targets), chunksize=8)
    return [image_io.load_gray(t) for t in targets]


def msu_pair(image: np.ndarray, rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    """(input, label) float32 (size, size) in about [-1, 1], as ImageFromFile_AutoEcoder.get_data."""
    h, w = image.shape
    top, left = rng.integers(h - size), rng.integers(w - size)
    label = image[top:top + size, left:left + size].astype(np.float32)
    noisy = (label + (rng.random() - 0.5) * 50) * (rng.random() + 0.5)   # brightness, then contrast
    noisy = noisy / 128.0 - 1 + rng.normal(0, 1, (size, size)).astype(np.float32)
    kernel = int(rng.integers(0, 4)) * 2 + 1
    return cv2.GaussianBlur(noisy, (kernel, kernel), 0), label / 128.0 - 1


class MsuPatches(Dataset):
    """Random MSU training pairs from in-memory STFT images; rng from (seed, epoch, worker, index)."""

    def __init__(self, images: list[np.ndarray], length: int, size: int, seed: int):
        self.images, self.length, self.size, self.seed, self.epoch = images, length, size, seed, 0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        worker = get_worker_info()
        rng = np.random.default_rng([self.seed, self.epoch, 0 if worker is None else worker.id, index])
        noisy, label = msu_pair(self.images[rng.integers(len(self.images))], rng, self.size)
        return torch.from_numpy(noisy)[None], torch.from_numpy(label)[None]


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="40 images, 20 tiny steps on CPU")
    args = parser.parse_args()
    train = dict(PARAMS["training"])
    out = HERE / ("checkpoints_smoke" if args.smoke else "checkpoints")
    pool = pd.read_csv(config.TRAIN_CLEAN_DIR / "pool.csv")
    val_meta = eval_sets.load_meta("val")
    if args.smoke:
        train.update(steps=20, batch_size=4, workers=2, log_every=5, checkpoint_every=10, val_patches=8)
        pool, val_meta = pool.head(40), val_meta.head(4)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(train["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = build_cache([config.TRAIN_CLEAN_DIR / f for f in pool["file"]], HERE / "cache" / "train", train["workers"])
    val_images = build_cache([eval_sets.triplet_path("val", p, "clean") for p in val_meta["probe_id"]],
                             HERE / "cache" / "val", train["workers"])
    val = MsuPatches(val_images, train["val_patches"], train["patch"], seed=train["seed"] + 1)
    val_x, val_y = (torch.stack(t).to(device) for t in zip(*(val[k] for k in range(len(val)))))
    data = MsuPatches(images, train["steps"] * train["batch_size"], train["patch"], train["seed"])
    loader = DataLoader(data, batch_size=train["batch_size"], num_workers=train["workers"],
                        pin_memory=device.type == "cuda")

    model = EnhancementAEC(PARAMS["model"]["nf"]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=train["lr"], betas=tuple(train["betas"]))
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, {len(images)} train images, device {device}")
    for step, (noisy, label) in enumerate(loader, start=1):
        model.train()
        loss = torch.mean((model(noisy.to(device)) - label.to(device)) ** 2)   # MSU: tf.nn.l2_loss (sum / 2)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % train["log_every"] == 0:
            append_csv(out / "log.csv", {"step": step, "loss": round(loss.item(), 6)})
            print(f"step {step}  loss {loss.item():.5f}", flush=True)
        if step % train["checkpoint_every"] == 0 or step == train["steps"]:
            model.eval()
            with torch.no_grad():
                v = torch.mean((model(val_x) - val_y) ** 2).item()
            append_csv(out / "val.csv", {"step": step, "val_loss": round(v, 6)})
            torch.save({"model": model.state_dict(), "step": step, "params": PARAMS}, out / f"step_{step:06d}.pt")
            torch.save({"model": model.state_dict(), "step": step, "params": PARAMS}, out / "last.pt")
            print(f"step {step}  val loss {v:.5f}  saved checkpoint", flush=True)


if __name__ == "__main__":
    main()
