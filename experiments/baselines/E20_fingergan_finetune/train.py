"""Fine-tune FingerGAN's released generator (E03's weights) on our normalised damaged -> clean pairs.

Usage (GPU):  FPR_DATA_VERSION=v3 python experiments/baselines/E20_fingergan_finetune/train.py
Quick local check:                 FPR_DATA_VERSION=v3 python experiments/baselines/E20_fingergan_finetune/train.py --smoke
Input = the MSU-normalised damaged 192 px window /255 (FingerGAN's window and input range), target = the
normalised clean window /255 (in its sigmoid range), from the stored v3 train set (E19's NormPairs).
No discriminator (the authors released only the generator): L1 over the window plus lam x clDice
(E07's cldice_term) inside the damage mask grown by dilate_px. Adam, constant lr after a short warm-up.
Writes checkpoints/.
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
from experiments.baselines.E03_fingergan.network import FingerGAN
from experiments.baselines.E19_msu_aec_norm.pairs import flatten, NormPairs, val_windows
from fpr.training.aux_losses import cldice_term, dilate

HERE = Path(__file__).resolve().parent
E03 = HERE.parent / "E03_fingergan"
PARAMS = json.loads((HERE / "params.json").read_text())


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def losses(net, degraded, clean, mask, train: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(total, L1, clDice) for one batch; images (B, 1, P, P) in [0, 1]."""
    pred = net(degraded)
    l1 = (pred - clean).abs().mean()
    m = dilate(mask, train["dilate_px"])
    cl = cldice_term(pred, clean, m).mean()
    return l1 + train["lam_cldice"] * cl, l1, cl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="a few tiny steps on CPU")
    args = parser.parse_args()
    assert config.DATA_VERSION == PARAMS["data_version"], "set FPR_DATA_VERSION=v3"
    train = dict(PARAMS["training"])
    out = HERE / ("checkpoints_smoke" if args.smoke else "checkpoints")
    if args.smoke:
        train.update(steps=4, batch_size=2, windows_per_sample=2, workers=2, log_every=2, checkpoint_every=2, val_patches=4)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(train["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = FingerGAN(in_channels=1).to(device)
    state = torch.load(E03 / PARAMS["weights"], map_location=device, weights_only=False)
    net.load_state_dict(state["UNet_model_state"])                      # released weights: the starting point
    k = train["windows_per_sample"]
    assert train["batch_size"] % k == 0, "batch_size counts windows: a multiple of windows_per_sample"
    data = NormPairs(PARAMS["window"], seed=train["seed"], windows=k)
    val = [t.to(device) for t in val_windows(train["val_patches"], PARAMS["window"], train["seed"] + 1)]
    optimiser = torch.optim.Adam(net.parameters(), lr=train["lr"])
    warmup = lambda step: min(1.0, (step + 1) / train["warmup_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, warmup)
    print(f"{sum(p.numel() for p in net.parameters()):,} parameters, {len(data):,} train samples, "
          f"device {device}", flush=True)

    def validate(step: int) -> None:
        net.eval()
        with torch.no_grad():
            parts = [losses(net, *(v[k:k + 16] for v in val), train) for k in range(0, len(val[0]), 16)]
        total, l1, cl = (sum(p[i].item() for p in parts) / len(parts) for i in range(3))
        append_csv(out / "val.csv", {"step": step, "val_loss": round(total, 6), "val_l1": round(l1, 6),
                                     "val_cldice": round(cl, 6)})
        print(f"step {step}  val loss {total:.5f}  (L1 {l1:.5f}, clDice {cl:.5f})", flush=True)

    validate(0)                                                          # the released weights, zero-shot
    step, epoch = 0, 0
    while step < train["steps"]:
        data.set_epoch(epoch)
        loader = DataLoader(data, batch_size=train["batch_size"] // k, shuffle=True, drop_last=True,
                            num_workers=train["workers"], pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(train["seed"] + epoch))
        for degraded, clean, mask in loader:
            degraded, clean, mask = flatten(degraded), flatten(clean), flatten(mask)
            step += 1
            net.train()
            total, l1, cl = losses(net, degraded.to(device), clean.to(device), mask.to(device), train)
            optimiser.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), train["grad_clip"])
            optimiser.step()
            scheduler.step()
            if step % train["log_every"] == 0:
                append_csv(out / "log.csv", {"step": step, "loss": round(total.item(), 6),
                                             "l1": round(l1.item(), 6), "cldice": round(cl.item(), 6)})
                print(f"step {step}  loss {total.item():.5f}  (L1 {l1.item():.5f}, clDice {cl.item():.5f})",
                      flush=True)
            if step % train["checkpoint_every"] == 0 or step == train["steps"]:
                validate(step)
                torch.save({"UNet_model_state": net.state_dict(), "step": step, "params": PARAMS}, out / "last.pt")
            if step >= train["steps"]:
                break
        epoch += 1


if __name__ == "__main__":
    main()
