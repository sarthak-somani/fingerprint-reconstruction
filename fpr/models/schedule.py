"""Adapted from zsyOAOA/ResShift (Yue, Wang, Loy, TPAMI 2024), S-Lab License 1.0.

Copyright 2022 S-Lab. Non-commercial use only; licence: github.com/zsyOAOA/ResShift/blob/master/LICENSE
The residual-shift noise schedule: how far x_t has moved from x_0 towards y_0 (eta_t)
and how much noise it carries (kappa^2 eta_t). Used by resshift.py and ccdf.py. Defaults come from
E02_resshift's params.json ('diffusion', config.REFERENCE_PARAMS), which every ResShift experiment shares.
Equation numbers refer to the TPAMI paper, arXiv 2403.07319v3.
"""
import json
import math

import numpy as np
import torch

import config

DIFFUSION = json.loads(config.REFERENCE_PARAMS.read_text())["diffusion"]


def sqrt_eta_schedule(steps: int = DIFFUSION["steps"], kappa: float = DIFFUSION["kappa"],
                      min_noise: float = DIFFUSION["min_noise"],
                      sqrt_eta_end: float = DIFFUSION["sqrt_eta_end"],
                      power: float = DIFFUSION["power"]) -> np.ndarray:
    """sqrt(eta_t) for t = 1..T, float64 (T,), increasing from sqrt(eta_1) to sqrt(eta_T).

    Eq. (12): sqrt(eta_t) = sqrt(eta_1) * b0 ** beta_t
    Eq. (13a): beta_t = ((t - 1) / (T - 1)) ** p * (T - 1)
    Eq. (13b): b0 = exp(log(eta_T / eta_1) / (2 (T - 1)))
    sqrt(eta_1) = min(min_noise / kappa, min_noise), as in ResShift's code, so the first
    step adds noise of std kappa * sqrt(eta_1) = min_noise (for kappa >= 1).
    """
    sqrt_eta_start = min(min_noise / kappa, min_noise)
    b0 = math.exp(math.log(sqrt_eta_end / sqrt_eta_start) / (steps - 1))     # = Eq. (13b) on sqrt
    beta = np.linspace(0, 1, steps) ** power * (steps - 1)                    # Eq. (13a)
    return sqrt_eta_start * b0 ** beta                                         # Eq. (12)


def build_schedule(sqrt_eta: np.ndarray | None = None, kappa: float = DIFFUSION["kappa"],
                   timesteps: list[int] | None = None) -> dict:
    """All per-step quantities, as float32 tensors of shape (T + 1,) indexed by t = 0..T.

    Index 0 is the clean image: eta_0 = 0. Entries:
      eta        eta_t                                    Eq. (2)
      alpha      alpha_t = eta_t - eta_{t-1}              text after Eq. (1)
      post_var   kappa^2 eta_{t-1} / eta_t * alpha_t      Eq. (7), Eq. (8)
      coef_xt    eta_{t-1} / eta_t                        Eq. (7), Eq. (9)
      coef_x0    alpha_t / eta_t                          Eq. (7), Eq. (9)
      model_t    timestep number given to the model (differs from t after respacing)
    plus 'kappa' (float) and 'steps' (int T).
    """
    sqrt_eta = sqrt_eta_schedule() if sqrt_eta is None else sqrt_eta
    eta = np.concatenate([[0.0], sqrt_eta ** 2])
    eta_prev = np.concatenate([[0.0], eta[:-1]])
    alpha = eta - eta_prev
    safe_eta = np.where(eta > 0, eta, 1.0)                  # t = 0 is never a source step
    tensors = {"eta": eta, "alpha": alpha,
               "post_var": kappa ** 2 * eta_prev / safe_eta * alpha,
               "coef_xt": eta_prev / safe_eta, "coef_x0": alpha / safe_eta,
               "model_t": np.arange(len(eta)) if timesteps is None else np.array([0] + timesteps)}
    schedule = {k: torch.tensor(v, dtype=torch.float32) for k, v in tensors.items()}
    schedule["model_t"] = schedule["model_t"].long()
    schedule.update(kappa=kappa, steps=len(eta) - 1)
    return schedule


def respace(schedule: dict, steps: int) -> dict:
    """A schedule with fewer sampling steps that reuses a model trained on `schedule`.

    As ResShift's respace.py: keep the steps int(T / steps * k) + 1 for k = 0..steps-1
    (always including t = 1), rebuild alpha and the posterior from the kept eta values
    (Eq. (7)-(9)), and give the model its original timestep numbers.
    """
    total = schedule["steps"]
    if steps == total:
        return schedule
    kept = sorted({int(total / steps * k) + 1 for k in range(steps)})
    sqrt_eta = schedule["eta"][kept].double().sqrt().numpy()
    return build_schedule(sqrt_eta, schedule["kappa"], timesteps=kept)
