"""Adapted from zsyOAOA/ResShift (Yue, Wang, Loy, TPAMI 2024), S-Lab License 1.0.

Copyright 2022 S-Lab. Non-commercial use only; licence: github.com/zsyOAOA/ResShift/blob/master/LICENSE
Residual shifting in pixel space: x_t moves from the clean print x_0 (t = 0) to the
damaged print y_0 (t = T) plus noise; the model predicts x_0. Equation numbers refer
to the TPAMI paper, arXiv 2403.07319v3. All images are (B, 1, H, W) float in [-1, 1].
The model is any callable model(x_t, t, cond) -> x0_hat, with t a (B,) long tensor.
"""
import torch

from fpr.models.schedule import build_schedule, DIFFUSION, respace


def at(values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """values[t] reshaped to (B, 1, 1, 1) for broadcasting; t is (B,) long."""
    return values.to(t.device)[t].view(-1, 1, 1, 1)


def q_sample(x0: torch.Tensor, y0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
             schedule: dict | None = None) -> torch.Tensor:
    """Sample x_t ~ q(x_t | x_0, y_0), Eq. (2): x_t = x_0 + eta_t (y_0 - x_0) + kappa sqrt(eta_t) noise.

    x0, y0, noise: (B, 1, H, W); t: (B,) long in 0..T. Returns (B, 1, H, W).
    The residual e_0 = y_0 - x_0 is zero outside the damage, so there x_t is x_0 plus noise.
    """
    schedule = schedule or build_schedule()
    eta = at(schedule["eta"], t)
    return x0 + eta * (y0 - x0) + schedule["kappa"] * eta.sqrt() * noise


def scale_input(x_t: torch.Tensor, t: torch.Tensor, schedule: dict) -> torch.Tensor:
    """x_t / (3 kappa sqrt(eta_t) + 1): keeps the model input near [-1, 1] at every t.

    Not in the paper; ResShift's code does this (normalize_input, pixel-space branch).
    """
    return x_t / (3 * schedule["kappa"] * at(schedule["eta"], t).sqrt() + 1)


def predict_x0(model, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, schedule: dict) -> torch.Tensor:
    """x0_hat = f_theta(x_t, y_0, t), the paper's x_0 predictor (text after Eq. (9) and Eq. (10)).

    The model sees the scaled x_t, its own timestep number and `cond` (B, C, H, W), which the
    caller builds, e.g. y_0, the mask and the orientation field stacked on channels.
    """
    return model(scale_input(x_t, t, schedule), at(schedule["model_t"], t).view(-1), cond)


def training_loss(model, x0: torch.Tensor, y0: torch.Tensor, mask: torch.Tensor, cond: torch.Tensor,
                  schedule: dict | None = None) -> torch.Tensor:
    """Mean of ||x0_hat - x_0||^2 at a random step t ~ U{1..T}: Eq. (11) without the
    perceptual term, and with w_t omitted from Eq. (10) (the paper reports this works better).

    x0, y0, mask: (B, 1, H, W); mask is 1 on damaged pixels. Pixels inside the mask are weighted
    by params.json's mask_loss_weight (1 = ResShift's plain loss; our addition otherwise).
    Returns a scalar tensor.
    """
    schedule = schedule or build_schedule()
    t = torch.randint(1, schedule["steps"] + 1, (x0.shape[0],), device=x0.device)
    x_t = q_sample(x0, y0, t, torch.randn_like(x0), schedule)
    x0_hat = predict_x0(model, x_t, t, cond, schedule)
    weight = 1 + (DIFFUSION["mask_loss_weight"] - 1) * mask
    return (weight * (x0_hat - x0) ** 2).sum() / weight.sum()


@torch.no_grad()
def sample(model, y0: torch.Tensor, mask: torch.Tensor, cond: torch.Tensor, steps: int | None = None,
           schedule: dict | None = None) -> torch.Tensor:
    """Reverse process, Algorithm 2: x_T ~ N(y_0, kappa^2 eta_T I), then for t = T..1
    x_{t-1} ~ N(mu, Sigma) with mu from Eq. (9) and Sigma from Eq. (8).

    y0, mask: (B, 1, H, W); cond: (B, C, H, W); steps: sampling steps (default T; fewer
    uses respace()). x0_hat is clipped to [-1, 1] each step, as in ResShift's code. Returns
    (B, 1, H, W) in [-1, 1], with y_0 kept outside the mask (our addition; the Reconstructor
    pastes back as well).
    """
    schedule = schedule or build_schedule()
    schedule = respace(schedule, steps or schedule["steps"])
    big_t = schedule["steps"]
    t = torch.full((y0.shape[0],), big_t, dtype=torch.long, device=y0.device)
    x = y0 + schedule["kappa"] * at(schedule["eta"], t).sqrt() * torch.randn_like(y0)
    for step in range(big_t, 0, -1):
        t = torch.full_like(t, step)
        x0_hat = predict_x0(model, x, t, cond, schedule).clamp(-1, 1)
        mean = at(schedule["coef_xt"], t) * x + at(schedule["coef_x0"], t) * x0_hat     # Eq. (9)
        x = mean + at(schedule["post_var"], t).sqrt() * torch.randn_like(x)             # Eq. (8)
    return mask * x.clamp(-1, 1) + (1 - mask) * y0
