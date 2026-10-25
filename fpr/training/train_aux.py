"""Train ResShift with extra ridge-preserving losses on the predicted x0_hat (fpr/training/aux_losses.py).

One trainer for every aux-loss experiment; only params.json differs:
    E06 minutiae-weighted L1 | E07 + clDice | E08 Laplacian edge + MS-SSIM/L1 mix      (raw prints, v2)
    E12 clDice + edge + mix | E13-E16 one or all terms dropped (MSU-normalised prints, v3) | E26 E12 on raw v3
Usage:
    python -m fpr.training.train_aux --exp E07_aux_minutiae_cldice --calibrate 20   # lams -> calibration.json
    python -m fpr.training.train_aux --exp E07_aux_minutiae_cldice                   # train (resumes last.pt)
    python -m fpr.training.train_aux --exp E07_aux_minutiae_cldice --smoke           # 20 tiny CPU steps
A lam of null in 'aux_losses' is taken from calibration.json: the lam that makes the term 20% of L_diff at
init. Data: the stored damaged train set of params.json's data_version, MSU-normalised copies when
output_normalized is true, plus the minutiae maps when the minutiae term is on. Writes to <exp>/checkpoints/:
step_XXXXXX.pt (EMA weights), last.pt (for resuming), log.csv (losses and every extra term), val.csv
(diffusion loss only, comparable with E02's) and run_info.json.
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: makes config/fpr importable

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from fpr import experiments
from fpr.data import datasets
from fpr.models import inputs, resshift, schedule as resshift_schedule
from fpr.training import aux_losses
from fpr.training.common import (append_csv, build_model, lr_factor, optimiser_step, parse_args, resume,
                                 val_batch)


def lam_of(setting) -> float | None:
    return setting["lam"] if isinstance(setting, dict) else setting


def resolved_aux_losses(params: dict, calibration: Path) -> dict:
    """params.json's 'aux_losses' with every null lam replaced by calibration.json's lam_for_20_percent."""
    if all(lam_of(s) is not None for s in params["aux_losses"].values()):
        return params["aux_losses"]
    assert calibration.exists(), f"lams are null in params.json: run --calibrate 20 first ({calibration})"
    calibrated = json.loads(calibration.read_text())["terms"]
    out = {}
    for name, setting in params["aux_losses"].items():
        lam = lam_of(setting)
        lam = calibrated[name]["lam_for_20_percent"] if lam is None else lam
        out[name] = dict(setting, lam=lam) if isinstance(setting, dict) else lam
    return out


def build_schedule(diffusion: dict) -> dict:
    """ResShift's schedule from this experiment's 'diffusion' settings (E02's code, our numbers)."""
    sqrt_eta = resshift_schedule.sqrt_eta_schedule(diffusion["steps"], diffusion["kappa"], diffusion["min_noise"],
                                                   diffusion["sqrt_eta_end"], diffusion["power"])
    return resshift_schedule.build_schedule(sqrt_eta, diffusion["kappa"])


def print_eta_table(schedule: dict, eta_max: float) -> None:
    """Which steps t get the extra terms (eta_t <= eta_max)."""
    eta = schedule["eta"]
    passing = [t for t in range(1, schedule["steps"] + 1) if eta[t] <= eta_max]
    print("t    eta_t   extra terms")
    for t in range(1, schedule["steps"] + 1):
        print(f"{t:<4} {eta[t].item():.4f}  {'yes' if t in passing else 'no'}")
    print(f"gate eta_t <= {eta_max}: t = {passing[0]}..{passing[-1]} ({len(passing)} of {schedule['steps']} steps)")


def training_losses(model, batch: dict, device: torch.device, schedule: dict, params: dict,
                    aux_losses_cfg: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """(loss, l_diff, logs) for one batch: loss = L_diff + the weighted extra terms.

    L_diff is resshift.training_loss written out, with the same random calls in the same order
    (t, then the noise), so with no extra terms the loss is E02's bit for bit. The extra terms
    use no random numbers, so the random streams (and the data order) stay E02's.
    """
    x0, y0, mask, cond = inputs.batch_inputs(batch, device)
    t = torch.randint(1, schedule["steps"] + 1, (x0.shape[0],), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = resshift.q_sample(x0, y0, t, noise, schedule)
    x0_hat = resshift.predict_x0(model, x_t, t, cond, schedule)
    weight = 1 + (params["diffusion"]["mask_loss_weight"] - 1) * mask
    l_diff = (weight * (x0_hat - x0) ** 2).sum() / weight.sum()
    minutiae_w = batch["minutiae_w"].to(device) if "minutiae_w" in batch else None
    eta_t = schedule["eta"].to(device)[t]
    aux_total, logs = aux_losses.aux_loss(x0_hat, x0, mask, eta_t, aux_losses_cfg, params["aux"], minutiae_w)
    return l_diff + aux_total, l_diff, logs


@torch.no_grad()
def val_loss(model, batch: dict, device, seed: int, schedule: dict, params: dict) -> float:
    """Diffusion loss only on the fixed val patches, with fixed t and noise (as E02's val.csv)."""
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        chunks = [{k: v[i:i + 32] for k, v in batch.items()} for i in range(0, len(batch["clean"]), 32)]
        losses = [training_losses(model, chunk, device, schedule, params, aux_losses_cfg={})[1].item()
                  for chunk in chunks]
    return float(np.mean(losses))


def normalized_data(params: dict) -> bool:
    """True for the experiments trained on MSU-normalised prints (E12-E16)."""
    return bool(params.get("output_normalized"))


def build_loader(params: dict, train: dict, device: torch.device) -> tuple[datasets.FixedTrainDataset, DataLoader]:
    """The stored damaged train set (+ minutiae maps if the minutiae term is on) and E02's shuffled loader."""
    wanted = params.get("data_version", "v2")
    assert config.DATA_VERSION == wanted, f"set FPR_DATA_VERSION={wanted}"
    minutiae_dir = config.TRAIN_MINUTIAE_DIR if "minutiae" in params["aux_losses"] else None   # script 09's maps
    data = datasets.FixedTrainDataset(seed=train["seed"], minutiae_dir=minutiae_dir, normalized=normalized_data(params))
    loader = DataLoader(data, batch_size=train["batch_size"], shuffle=True, drop_last=True,
                        num_workers=train["workers"], pin_memory=device.type == "cuda",
                        persistent_workers=train["workers"] > 0)
    return data, loader


def calibrate(params: dict, calibration: Path, n_batches: int, device: torch.device) -> None:
    """Forward + backward of the untrained model on n_batches batches, no optimiser step.

    Reports each extra term against L_diff and the lam that would make it 20% of L_diff
    (every term measured with lam 1). Writes only calibration.json.
    """
    train = dict(params["training"], workers=min(params["training"]["workers"], 8))
    unit_lams = {name: dict(s, lam=1.0) if isinstance(s, dict) else 1.0 for name, s in params["aux_losses"].items()}
    torch.manual_seed(train["seed"])
    model = build_model(params, train, device)[0]
    _, loader = build_loader(params, train, device)
    schedule = build_schedule(params["diffusion"])
    rows, first_ids = [], None
    for batch in loader:
        first_ids = first_ids or list(batch["sample_id"])
        start = time.time()
        model.zero_grad(set_to_none=True)
        loss, l_diff, logs = training_losses(model, batch, device, schedule, params, aux_losses_cfg=unit_lams)
        loss.backward()
        grads_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        rows.append({"loss": loss.item(), "l_diff": l_diff.item(), **{k: v.item() for k, v in logs.items()},
                     "finite": math.isfinite(loss.item()) and grads_finite, "sec": time.time() - start})
        print(f"batch {len(rows)}/{n_batches}  l_diff {l_diff.item():.4f}  "
              + "  ".join(f"{k} {v.item():.4f}" for k, v in logs.items()), flush=True)
        if len(rows) == n_batches:
            break

    l_diff = float(np.mean([r["l_diff"] for r in rows]))
    terms = {}
    for name in params["aux_losses"]:
        value = float(np.mean([r[name] for r in rows if r["n_gated"] > 0]))
        terms[name] = {"raw_mean_over_gated": value, "lam_for_20_percent": float(f"{0.2 * l_diff / value:.2g}")}
    result = {"batches": len(rows), "batch_size": train["batch_size"], "device": str(device),
              "data_version": config.DATA_VERSION, "l_diff_mean": l_diff,
              "mean_n_gated": float(np.mean([r["n_gated"] for r in rows])), "terms": terms,
              "all_finite": all(r["finite"] for r in rows),
              "sec_per_step": float(np.mean([r["sec"] for r in rows])),
              "aux_losses": params["aux_losses"], "aux": params["aux"], "first_batch_sample_ids": first_ids}
    calibration.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nL_diff {l_diff:.5f}  mean gated per batch {result['mean_n_gated']:.1f}/{train['batch_size']}  "
          f"finite {result['all_finite']}  {result['sec_per_step']:.2f} s/step on {device}")
    print(f"{'term':<10} {'raw':>9} {'lam for 20%':>12}")
    for name, row in terms.items():
        print(f"{name:<10} {row['raw_mean_over_gated']:>9.5f} {row['lam_for_20_percent']:>12.2g}")
    print(f"wrote {calibration}")


def log_row(step: int, sums: dict, count: int, lr: float, seconds: float, device: torch.device) -> dict:
    """One log.csv row: the losses and extra terms averaged over the last `count` steps."""
    means = {k: v.item() / count for k, v in sums.items()}
    bad = [k for k, v in means.items() if not math.isfinite(v)]
    if bad:
        raise RuntimeError(f"step {step}: non-finite {bad}; stopping")
    gpu_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    return {"step": step, **{k: round(v, 6) for k, v in means.items()}, "lr": lr,
            "sec_per_step": round(seconds, 4), "gpu_mem_gb": round(gpu_gb, 2)}


def train_model(params: dict, train: dict, out: Path, calibration: Path, val_seed: int | None = None) -> None:
    """The training loop. val_seed fixes the validation patches and draws (default: training.seed)."""
    val_seed = train["seed"] if val_seed is None else val_seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = build_schedule(params["diffusion"])
    aux_cfg = resolved_aux_losses(params, calibration)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(train["seed"])
    use_amp = train["amp"] and device.type == "cuda"

    model, ema, optimiser, scaler = build_model(params, train, device)
    step = resume(out, model, ema, optimiser, scaler, device)
    data, loader = build_loader(params, train, device)
    val = val_batch(train["val_patches"], val_seed, normalized=normalized_data(params))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params / 1e6:.2f}M parameters, {len(data)} train samples, device {device}, amp {use_amp}, "
          f"extra terms {aux_cfg}")

    tick, sums, count = time.time(), {}, 0
    has_run_info = (out / "run_info.json").exists()
    while step < train["steps"]:
        data.set_epoch(step // max(1, len(loader)))
        for batch in loader:
            if not has_run_info:                               # the first run's settings and first batch
                (out / "run_info.json").write_text(json.dumps(
                    {"data_version": config.DATA_VERSION, "params": params, "aux_losses": aux_cfg,
                     "start_step": step, "first_batch_sample_ids": list(batch["sample_id"])}, indent=2) + "\n")
                has_run_info = True
            for group in optimiser.param_groups:
                group["lr"] = train["lr"] * lr_factor(step, train)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss, l_diff, logs = training_losses(model, batch, device, schedule, params, aux_cfg)
            optimiser_step(loss, model, ema, optimiser, scaler, train)
            step += 1
            for key, value in {"loss": loss.detach(), "l_diff": l_diff.detach(), **logs}.items():
                sums[key] = sums.get(key, 0) + value.float()  # summed on the device: no sync per step
            count += 1

            if step % train["log_every"] == 0:
                row = log_row(step, sums, count, optimiser.param_groups[0]["lr"], (time.time() - tick) / count, device)
                append_csv(out / "log.csv", row)
                print("  ".join(f"{k} {v}" for k, v in row.items()), flush=True)
                tick, sums, count = time.time(), {}, 0
            if step % train["checkpoint_every"] == 0 or step == train["steps"]:
                v = val_loss(ema, val, device, val_seed, schedule, params)
                append_csv(out / "val.csv", {"step": step, "val_loss": round(v, 6)})
                torch.save({"ema": ema.state_dict(), "step": step, "params": params,
                            "data_version": config.DATA_VERSION}, out / f"step_{step:06d}.pt")
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimiser": optimiser.state_dict(),
                            "scaler": scaler.state_dict(), "step": step, "params": params,
                            "data_version": config.DATA_VERSION}, out / "last.pt")
                print(f"step {step}  val loss {v:.5f}  saved checkpoint", flush=True)
            if step >= train["steps"]:
                break


def main() -> None:
    def extra(parser):
        parser.add_argument("--calibrate", type=int, nargs="?", const=20, default=None, metavar="N",
                            help="report the extra terms against L_diff on N batches (default 20), then stop")
    args, params, train, out = parse_args(__doc__, extra)
    calibration = experiments.experiment_dir(args.exp) / "calibration.json"
    print_eta_table(build_schedule(params["diffusion"]), params["aux"]["eta_max"])
    if args.calibrate:
        calibrate(params, calibration, args.calibrate, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return
    train_model(params, train, out, calibration)


if __name__ == "__main__":
    main()
