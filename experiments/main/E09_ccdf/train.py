"""E09_ccdf 'training': the regression control, then CCDF's val sweep on E02's model.

Usage (GPU; scripts/slurm/submit_chain.sh E09_ccdf <partition>):  python experiments/main/E09_ccdf/train.py
1. Train the deterministic regression control (E02's U-Net, x_t = y_0, t = 0, L1), resumable.
2. Wait for E02_resshift's v2 run to save its final step (the two can train in parallel).
3. Sweep starting estimate x t0 x projection on val (fpr/validation/ccdf_sweep.py) -> checkpoints/sweep.csv, and write
   the selected configuration to selected.json, which reconstructor.py uses for val and test.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: makes config/fpr importable

import torch
from torch.utils.data import DataLoader

import config
from experiments.main.E09_ccdf.reconstructor import build_reconstructor, E02_PARAMS
from fpr.data import datasets
from fpr.models import ccdf, inputs
from fpr.models.unet import UNet
from fpr.training.common import append_csv, lr_factor, val_batch
from fpr.validation import ccdf_sweep as sweep

HERE = Path(__file__).resolve().parent
PARAMS = json.loads((HERE / "params.json").read_text())
OUT = HERE / "checkpoints"


def train_regressor(train: dict, device: torch.device) -> UNet:
    """Stage 1: E02's U-Net as a deterministic x_0 regressor (L1), EMA weights returned."""
    model = UNet(**E02_PARAMS["model"]).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimiser = torch.optim.AdamW(model.parameters(), lr=train["lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=train["amp"] and device.type == "cuda")
    step, last = 0, OUT / "regressor_last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(state["model"]), ema.load_state_dict(state["ema"])
        optimiser.load_state_dict(state["optimiser"]), scaler.load_state_dict(state["scaler"])
        step = state["step"]
    data = datasets.FixedTrainDataset(seed=train["seed"])
    loader = DataLoader(data, batch_size=train["batch_size"], shuffle=True, drop_last=True,
                        num_workers=train["workers"], pin_memory=True, persistent_workers=train["workers"] > 0)
    val = val_batch(train["val_patches"], train["seed"])
    while step < train["steps"]:
        data.set_epoch(step // len(loader))
        for batch in loader:
            for group in optimiser.param_groups:
                group["lr"] = train["lr"] * lr_factor(step, train)
            x0, y0, _, cond = inputs.batch_inputs(batch, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                loss = (model(y0, torch.zeros(len(y0), dtype=torch.long, device=device), cond) - x0).abs().mean()
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
                append_csv(OUT / "regressor_log.csv", {"step": step, "l1": round(loss.item(), 6)})
            if step % train["checkpoint_every"] == 0 or step == train["steps"]:
                x0v, y0v, _, condv = inputs.batch_inputs(val, device)
                v = (ccdf.regress(ema, y0v, condv) - x0v).abs().mean().item()
                append_csv(OUT / "regressor_val.csv", {"step": step, "val_l1": round(v, 6)})
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimiser": optimiser.state_dict(),
                            "scaler": scaler.state_dict(), "step": step}, last)
                print(f"regressor step {step}  val L1 {v:.5f}", flush=True)
            if step >= train["steps"]:
                break
    return ema


def wait_for_e02(e02: dict) -> Path:
    """Block until E02's checkpoint holds its final step (v2 run), up to wait_hours."""
    path, deadline = config.REPO_ROOT / e02["checkpoint"], time.time() + 3600 * e02["wait_hours"]
    while True:
        step = torch.load(path, map_location="cpu", weights_only=True)["step"] if path.exists() else -1
        if step >= e02["final_step"]:
            print(f"E02 checkpoint at step {step}", flush=True)
            return path
        assert time.time() < deadline, f"E02 still at step {step} after {e02['wait_hours']} h"
        time.sleep(300)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(PARAMS["regressor"]["seed"])
    regressor = train_regressor(PARAMS["regressor"], device)
    e02_checkpoint = wait_for_e02(PARAMS["e02"])
    method = build_reconstructor(PARAMS, config=sweep.PLAIN_E02, regressor=regressor, checkpoint=e02_checkpoint)
    table = sweep.run_sweep(method, sweep.configurations(PARAMS["sweep"]), OUT, PARAMS["sweep"]["workers"])
    selected = sweep.select(table)
    (HERE / PARAMS["inference"]["selected"]).write_text(json.dumps(selected, indent=2) + "\n")
    print("selected", selected, flush=True)


if __name__ == "__main__":
    main()
