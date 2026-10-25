"""Train E10_ctmsr: CTMSR Stage 1, consistency training (consistency.py), on the stored damaged train set.

Usage (via scripts/slurm/submit_chain.sh E10_ctmsr <partition>):  python experiments/one_step/E10_ctmsr/train.py
Writes checkpoints/: last.pt (model, EMA, optimiser; resumes), step_XXXXXX.pt (EMA), log.csv
(CT loss, intervals T(k)) and val.csv (one-step L1 of the EMA model on fixed val patches).
E11_ctmsr_dtm imports `make_loader`, `one_step_val` and `save` from here.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: makes config/fpr importable

import lpips
import torch
from torch.utils.data import DataLoader

from experiments.one_step.E10_ctmsr import consistency
from fpr.data import datasets
from fpr.models import inputs
from fpr.models.unet import UNet
from fpr.training.common import append_csv, val_batch

HERE = Path(__file__).resolve().parent
PARAMS = json.loads((HERE / "params.json").read_text())


def make_loader(train: dict) -> tuple[datasets.FixedTrainDataset, DataLoader]:
    data = datasets.FixedTrainDataset(seed=train["seed"])
    return data, DataLoader(data, batch_size=train["batch_size"], shuffle=True, drop_last=True,
                            num_workers=train["workers"], pin_memory=True, persistent_workers=train["workers"] > 0)


@torch.no_grad()
def one_step_val(net, val: dict, sigma_max: float, device, seed: int) -> float:
    """L1 of the one-step output on fixed val patches, fixed noise (comparable across checkpoints)."""
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        x0, y0, _, cond = inputs.batch_inputs(val, device)
        out = torch.cat([consistency.one_step(net, y0[i:i + 32], cond[i:i + 32], sigma_max)
                         for i in range(0, len(y0), 32)])
    return (out - x0).abs().mean().item()


def save(path: Path, model, ema, optimiser, scaler, step: int) -> None:
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimiser": optimiser.state_dict(),
                "scaler": scaler.state_dict(), "step": step, "params": PARAMS}, path)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    train, cons, out = PARAMS["training"], PARAMS["consistency"], HERE / "checkpoints"
    out.mkdir(exist_ok=True)
    torch.manual_seed(train["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = train["amp"] and device.type == "cuda"
    model = UNet(**PARAMS["model"]).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    lpips_net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)
    optimiser = torch.optim.AdamW(model.parameters(), lr=train["lr"], betas=tuple(train["betas"]), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step = 0
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["model"]), ema.load_state_dict(state["ema"])
        optimiser.load_state_dict(state["optimiser"]), scaler.load_state_dict(state["scaler"])
        step = state["step"]
        print(f"resumed at step {step}")
    data, loader = make_loader(train)
    val = val_batch(train["val_patches"], train["seed"])
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, {len(data)} samples, amp {amp}")

    tick = time.time()
    while step < train["steps"]:
        data.set_epoch(step // len(loader))
        for batch in loader:
            x0, y0, _, cond = inputs.batch_inputs(batch, device)
            intervals = consistency.num_intervals(step, train["steps"], cons["s0"], cons["s1"])
            loss, _ = consistency.ct_step(model, lpips_net, x0, y0, cond, intervals, cons["sigma_max"], amp)
            assert torch.isfinite(loss), f"non-finite CT loss at step {step}"
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip"])
            scaler.step(optimiser), scaler.update()
            with torch.no_grad():
                for p_ema, p in zip(ema.parameters(), model.parameters()):
                    p_ema.lerp_(p, 1 - train["ema_decay"])
            step += 1
            if step % train["log_every"] == 0:
                seconds = (time.time() - tick) / train["log_every"]
                append_csv(out / "log.csv", {"step": step, "ct_loss": round(loss.item(), 6), "intervals": intervals,
                                             "sec_per_step": round(seconds, 4)})
                print(f"step {step}  CT loss {loss.item():.5f}  T {intervals}  {seconds:.3f} s/step", flush=True)
                tick = time.time()
            if step % train["checkpoint_every"] == 0 or step == train["steps"]:
                v = one_step_val(ema, val, cons["sigma_max"], device, train["seed"])
                append_csv(out / "val.csv", {"step": step, "val_one_step_l1": round(v, 6)})
                torch.save({"ema": ema.state_dict(), "step": step, "params": PARAMS}, out / f"step_{step:06d}.pt")
                save(out / "last.pt", model, ema, optimiser, scaler, step)
                print(f"step {step}  val one-step L1 {v:.5f}  saved", flush=True)
            if step >= train["steps"]:
                break


if __name__ == "__main__":
    main()
