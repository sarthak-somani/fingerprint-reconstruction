"""CCDF (Chung, Sim & Ye, Come-Closer-Diffuse-Faster, CVPR 2022) on ResShift's chain.

Instead of starting the reverse process at t = T from y_0 plus noise, start at t0 < T from a
starting estimate x_init, forward-diffused in one jump with ResShift's kernel (CCDF Eq. 12 with
ResShift's Eq. 2): x_t0 = x_init + eta_t0 (y_0 - x_init) + kappa sqrt(eta_t0) eps. Then run E02's
reverse steps t0 -> 1 (ResShift Eq. 8-9). With `project`, every step is followed by CCDF's
data-consistency step (Eq. 13-14): outside the mask the pixels are reset to their known forward
marginal, y_0 + kappa sqrt(eta_{t-1}) eps (the residual is 0 there), which is non-expansive.
All images are (B, 1, H, W) in [-1, 1]; mask is 1 on damaged pixels.
"""
import torch

from fpr.models.resshift import at, predict_x0


def jump(x_init: torch.Tensor, y0: torch.Tensor, t0: int, schedule: dict) -> torch.Tensor:
    """x_t0 ~ q(x_t0 | x_init, y_0) in one step (no network call)."""
    t = torch.full((y0.shape[0],), t0, dtype=torch.long, device=y0.device)
    eta = at(schedule["eta"], t)
    return x_init + eta * (y0 - x_init) + schedule["kappa"] * eta.sqrt() * torch.randn_like(y0)


@torch.no_grad()
def e02_one_call(model, y0: torch.Tensor, cond: torch.Tensor, schedule: dict) -> torch.Tensor:
    """E02's deterministic x0 estimate: one network call at t = T on x_T = y_0 (no noise)."""
    t = torch.full((y0.shape[0],), schedule["steps"], dtype=torch.long, device=y0.device)
    return predict_x0(model, y0, t, cond, schedule).clamp(-1, 1)


@torch.no_grad()
def regress(net, y0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """The regression control: E02's U-Net with x_t = y_0 and t = 0, predicting x_0 directly."""
    t = torch.zeros(y0.shape[0], dtype=torch.long, device=y0.device)
    return net(y0, t, cond).clamp(-1, 1)


@torch.no_grad()
def sample(model, y0: torch.Tensor, mask: torch.Tensor, cond: torch.Tensor, x_init: torch.Tensor,
           t0: int, project: bool, schedule: dict) -> torch.Tensor:
    """CCDF reverse chain from t0 (0 = return x_init). y_0 is kept outside the mask at the end."""
    x = x_init if t0 == 0 else jump(x_init, y0, t0, schedule)
    t = torch.full((y0.shape[0],), 1, dtype=torch.long, device=y0.device)
    for step in range(t0, 0, -1):
        t = torch.full_like(t, step)
        x0_hat = predict_x0(model, x, t, cond, schedule).clamp(-1, 1)
        mean = at(schedule["coef_xt"], t) * x + at(schedule["coef_x0"], t) * x0_hat     # ResShift Eq. 9
        x = mean + at(schedule["post_var"], t).sqrt() * torch.randn_like(x)             # ResShift Eq. 8
        if project:
            known = y0 + schedule["kappa"] * at(schedule["eta"], t - 1).sqrt() * torch.randn_like(y0)
            x = mask * x + (1 - mask) * known                                            # CCDF Eq. 13-14
    return mask * x.clamp(-1, 1) + (1 - mask) * y0


def starting_estimate(kind: str, model, regressor, y0: torch.Tensor, cond: torch.Tensor,
                      schedule: dict) -> torch.Tensor:
    """x_init for one of params.json's 'inits': 'regressor', 'e02_x0' (= 'one_call', the name the v3
    experiments use) or 'y0'."""
    if kind == "regressor":
        return regress(regressor, y0, cond)
    if kind in ("e02_x0", "one_call"):
        return e02_one_call(model, y0, cond, schedule)
    assert kind == "y0", kind
    return y0
