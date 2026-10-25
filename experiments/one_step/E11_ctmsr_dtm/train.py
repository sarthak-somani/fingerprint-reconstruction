"""Train E11_ctmsr_dtm: CTMSR Stage 2, fine-tune E10's model with L_CT + 1.6 L_DTM (supp. Alg. 2-3).

Usage (via scripts/slurm/submit_chain.sh E11_ctmsr_dtm <partition>, after E10's job):
    python experiments/one_step/E11_ctmsr_dtm/train.py
Starts from E10's last.pt (online and EMA weights). A frozen copy of the online model, refreshed
every `refresh_every` iterations, defines both PF-ODE trajectories. Writes checkpoints/last.pt,
log.csv and val.csv (one-step L1 of the EMA model, comparable with E10's val.csv).
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: makes config/fpr importable

import lpips
import torch

import config
from experiments.one_step.E10_ctmsr import consistency
from experiments.one_step.E10_ctmsr.train import make_loader, one_step_val
from fpr.models import inputs
from fpr.models.unet import UNet
from fpr.training.common import append_csv, val_batch

HERE = Path(__file__).resolve().parent
REPO = config.REPO_ROOT
PARAMS = json.loads((HERE / "params.json").read_text())


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    train, dtm, cons, out = PARAMS["training"], PARAMS["dtm"], PARAMS["consistency"], HERE / "checkpoints"
    out.mkdir(exist_ok=True)
    torch.manual_seed(train["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = train["amp"] and device.type == "cuda"
    state = torch.load(REPO / PARAMS["init"], map_location=device, weights_only=True)
    print(f"starting from E10 step {state['step']}")
    model, ema = UNet(**PARAMS["model"]).to(device), UNet(**PARAMS["model"]).to(device)
    model.load_state_dict(state["model"]), ema.load_state_dict(state["ema"])
    ema.eval().requires_grad_(False)
    lpips_net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)
    optimiser = torch.optim.AdamW(model.parameters(), lr=dtm["lr"], betas=tuple(train["betas"]), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    sigmas, alphas = consistency.grid(cons["s1"], cons["sigma_max"], device)
    sigma, alpha = float(sigmas[dtm["grid_index"]]), float(alphas[dtm["grid_index"]])
    data, loader = make_loader(train)
    val = val_batch(train["val_patches"], train["seed"])
    append_csv(out / "val.csv", {"step": 0, "val_one_step_l1": round(one_step_val(ema, val, cons["sigma_max"], device, 0), 6)})

    step, frozen = 0, None
    while step < dtm["iterations"]:
        data.set_epoch(step // len(loader))
        for batch in loader:
            if step % dtm["refresh_every"] == 0:
                frozen = copy.deepcopy(model).eval().requires_grad_(False)
            x0, y0, _, cond = inputs.batch_inputs(batch, device)
            l_ct, x_hat0 = consistency.ct_step(model, lpips_net, x0, y0, cond, cons["s1"], cons["sigma_max"], amp)
            l_dtm = consistency.dtm_loss(frozen, lpips_net, x_hat0, x0, y0, cond, sigma, alpha, dtm["lambda_dtm"], amp)
            loss = dtm["lambda_ct"] * l_ct + l_dtm                       # lambda_dtm is inside dtm_loss
            assert torch.isfinite(loss), f"non-finite loss at step {step}"
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
                append_csv(out / "log.csv", {"step": step, "ct_loss": round(l_ct.item(), 6), "dtm_loss": round(l_dtm.item(), 6)})
                print(f"step {step}  CT {l_ct.item():.5f}  DTM {l_dtm.item():.5f}", flush=True)
            if step >= dtm["iterations"]:
                break
    v = one_step_val(ema, val, cons["sigma_max"], device, 0)
    append_csv(out / "val.csv", {"step": step, "val_one_step_l1": round(v, 6)})
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "step": step, "params": PARAMS}, out / "last.pt")
    print(f"done: {step} DTM iterations, val one-step L1 {v:.5f}", flush=True)


if __name__ == "__main__":
    main()
