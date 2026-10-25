"""Train a ResShift model (U-Net + residual-shifting diffusion) with the plain diffusion loss: E02_resshift.

Usage:  python -m fpr.training.train_resshift --exp E02_resshift            (GPU, ~12 h on an A40)
        python -m fpr.training.train_resshift --exp E02_resshift --smoke    (20 tiny CPU steps)
Settings come from the experiment's params.json ('training', 'model', 'diffusion'). Data: the stored damaged
v2 train set (FixedTrainDataset). Loss: resshift.training_loss (x0-prediction MSE at a random step t).
Writes to <exp>/checkpoints/: step_XXXXXX.pt (EMA weights, every checkpoint_every steps), last.pt (everything,
for resuming), log.csv (training loss) and val.csv (fixed validation loss on val people's patches).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: makes config/fpr importable

import numpy as np
import torch
from torch.utils.data import DataLoader

from fpr.data import datasets
from fpr.models import inputs, resshift
from fpr.training.common import (append_csv, build_model, lr_factor, optimiser_step, parse_args, resume,
                                 val_batch)


@torch.no_grad()
def val_loss(model, batch: dict, device, seed: int) -> float:
    """Diffusion loss on the fixed val patches, with fixed t and noise (comparable across steps)."""
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        x0, y0, mask, cond = inputs.batch_inputs(batch, device)
        losses = [resshift.training_loss(model, x0[i:i + 32], y0[i:i + 32], mask[i:i + 32],
                                         cond[i:i + 32]).item() for i in range(0, len(x0), 32)]
    return float(np.mean(losses))


def train_model(params: dict, train: dict, out: Path, val_seed: int | None = None) -> None:
    """The training loop. val_seed fixes the validation patches and draws (default: training.seed)."""
    val_seed = train["seed"] if val_seed is None else val_seed
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(train["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = train["amp"] and device.type == "cuda"

    model, ema, optimiser, scaler = build_model(params, train, device)
    step = resume(out, model, ema, optimiser, scaler, device)
    data = datasets.FixedTrainDataset(seed=train["seed"])
    loader = DataLoader(data, batch_size=train["batch_size"], shuffle=True, drop_last=True,
                        num_workers=train["workers"], pin_memory=device.type == "cuda",
                        persistent_workers=train["workers"] > 0)
    val = val_batch(train["val_patches"], val_seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params / 1e6:.2f}M parameters, {len(data)} train samples, device {device}, amp {use_amp}")

    tick = time.time()
    while step < train["steps"]:
        data.set_epoch(step // max(1, len(loader)))
        for batch in loader:
            for group in optimiser.param_groups:
                group["lr"] = train["lr"] * lr_factor(step, train)
            x0, y0, mask, cond = inputs.batch_inputs(batch, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = resshift.training_loss(model, x0, y0, mask, cond)
            optimiser_step(loss, model, ema, optimiser, scaler, train)
            step += 1

            if step % train["log_every"] == 0:
                seconds = (time.time() - tick) / train["log_every"]
                append_csv(out / "log.csv", {"step": step, "loss": round(loss.item(), 6),
                                             "lr": optimiser.param_groups[0]["lr"], "sec_per_step": round(seconds, 4)})
                print(f"step {step}  loss {loss.item():.5f}  {seconds:.3f} s/step", flush=True)
                tick = time.time()
            if step % train["checkpoint_every"] == 0 or step == train["steps"]:
                v = val_loss(ema, val, device, val_seed)
                append_csv(out / "val.csv", {"step": step, "val_loss": round(v, 6)})
                torch.save({"ema": ema.state_dict(), "step": step, "params": params}, out / f"step_{step:06d}.pt")
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimiser": optimiser.state_dict(),
                            "scaler": scaler.state_dict(), "step": step, "params": params}, out / "last.pt")
                print(f"step {step}  val loss {v:.5f}  saved checkpoint", flush=True)
            if step >= train["steps"]:
                break


def main() -> None:
    args, params, train, out = parse_args(__doc__)
    print(json.dumps({"exp": args.exp, "out": str(out), "training": train}))
    train_model(params, train, out)


if __name__ == "__main__":
    main()
