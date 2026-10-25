"""Train E19_msu_aec_norm: E05's MSU autoencoder on OUR damaged -> clean pairs, in the normalised domain.

Usage (GPU):  FPR_DATA_VERSION=v3 python experiments/baselines/E19_msu_aec_norm/train.py
Quick local check:                 FPR_DATA_VERSION=v3 python experiments/baselines/E19_msu_aec_norm/train.py --smoke
Input = the MSU-normalised damaged window, target = the normalised clean window (both mapped to [-1, 1],
the network's tanh range), from the stored v3 train set (NormPairs: script 10's copies, 113,544 samples).
Optimiser and loss are MSU's (Adam lr 1e-4, betas (0.5, 0.9), L2 over the whole window), as E05; only the
data changes: our contaminants instead of MSU's generic noise on STFT images. Writes checkpoints/.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: makes config/fpr importable

import torch
from torch.utils.data import DataLoader

import config
from experiments.baselines.E05_msu_aec.model import EnhancementAEC
from experiments.baselines.E19_msu_aec_norm.pairs import flatten, NormPairs, val_windows

HERE = Path(__file__).resolve().parent
PARAMS = json.loads((HERE / "params.json").read_text())


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def to_model(x: torch.Tensor) -> torch.Tensor:
    """[0, 1] -> [-1, 1] (grey 128 -> about 0)."""
    return x * 2 - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="20 tiny steps on CPU")
    args = parser.parse_args()
    assert config.DATA_VERSION == PARAMS["data_version"], "set FPR_DATA_VERSION=v3"
    train = dict(PARAMS["training"])
    out = HERE / ("checkpoints_smoke" if args.smoke else "checkpoints")
    if args.smoke:
        train.update(steps=20, batch_size=4, windows_per_sample=2, workers=2, log_every=5, checkpoint_every=10, val_patches=8)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(train["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k = train["windows_per_sample"]
    assert train["batch_size"] % k == 0, "batch_size counts windows: a multiple of windows_per_sample"
    data = NormPairs(train["patch"], seed=train["seed"], windows=k)
    val_x, val_y, _ = (t.to(device) for t in val_windows(train["val_patches"], train["patch"], train["seed"] + 1))
    model = EnhancementAEC(PARAMS["model"]["nf"]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=train["lr"], betas=tuple(train["betas"]))
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, {len(data):,} train samples, "
          f"device {device}", flush=True)

    step, epoch = 0, 0
    while step < train["steps"]:
        data.set_epoch(epoch)
        loader = DataLoader(data, batch_size=train["batch_size"] // k, shuffle=True, drop_last=True,
                            num_workers=train["workers"], pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(train["seed"] + epoch))
        for degraded, clean, _ in loader:
            degraded, clean = flatten(degraded), flatten(clean)
            step += 1
            model.train()
            loss = torch.mean((model(to_model(degraded.to(device))) - to_model(clean.to(device))) ** 2)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            if step % train["log_every"] == 0:
                append_csv(out / "log.csv", {"step": step, "loss": round(loss.item(), 6)})
                print(f"step {step}  loss {loss.item():.5f}", flush=True)
            if step % train["checkpoint_every"] == 0 or step == train["steps"]:
                model.eval()
                with torch.no_grad():
                    v = torch.mean((model(to_model(val_x)) - to_model(val_y)) ** 2).item()
                append_csv(out / "val.csv", {"step": step, "val_loss": round(v, 6)})
                state = {"model": model.state_dict(), "step": step, "params": PARAMS}
                torch.save(state, out / f"step_{step:06d}.pt")
                torch.save(state, out / "last.pt")
                print(f"step {step}  val loss {v:.5f}  saved checkpoint", flush=True)
            if step >= train["steps"]:
                break
        epoch += 1


if __name__ == "__main__":
    main()
