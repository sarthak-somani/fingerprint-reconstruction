"""CTMSR (You et al., Consistency Trajectory Matching for One-Step Generative SR, ICCV 2025) in pixel space.

Re-implemented from the paper (arXiv 2503.20349v5) and the authors' code (CVL-UESTC/CTMSR, MIT):
- forward process (Eq. 7): x_t = x_0 + alpha(t) e_0 + sigma(t) eps, e_0 = y_0 - x_0, with linear
  schedules sigma = sigma_max t/T, alpha = t/T (supp. A.1, rho = 1), on T + 1 grid points;
- step curriculum (supp. Eq. 24): T(k) = max(s0 - floor(k / K'), s1), K' = floor(K / (s0 - s1 + 1));
- consistency function (Eq. 9) with the code's EDM-style preconditioning: c_skip = sd^2 / ((s/0.1)^2 + sd^2),
  c_out = (s/0.1) / sqrt((s/0.1)^2 + sd^2), sd = 0.5, c_in = 1, so f = x at sigma = 0;
- CT loss (Eq. 10): d(f_theta(x_t), stopgrad f_theta(x_{t-1})), one t for the batch, the same eps
  for both points, d = 0.5 LPIPS-VGG + 0.5 Charbonnier (Eq. 20);
- DTM loss (supp. Alg. 3) for E11. One-step inference: f(y_0 + sigma_max eps, sigma_max).
The network F is E02's U-Net (same inputs as E02); its time input is 1000 sigma.
Images are (B, 1, H, W) in [-1, 1].
"""
import math

import torch

SIGMA_DATA = 0.5


def num_intervals(k: int, total: int, s0: int, s1: int) -> int:
    """T(k): trajectory intervals at training iteration k (decreasing curriculum, supp. Eq. 24)."""
    k_prime = math.floor(total / (s0 - s1 + 1))
    return max(s0 - math.floor(k / k_prime), s1)


def grid(intervals: int, sigma_max: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(sigmas, alphas), each (intervals + 1,), linear from 0 to sigma_max and from 0 to 1."""
    steps = torch.linspace(0, 1, intervals + 1, device=device)
    return sigma_max * steps, steps


def q_sample(x0: torch.Tensor, y0: torch.Tensor, sigma: torch.Tensor, alpha: torch.Tensor,
             noise: torch.Tensor) -> torch.Tensor:
    """Eq. 7; sigma, alpha broadcast as (B, 1, 1, 1) or scalars."""
    return x0 + alpha * (y0 - x0) + sigma * noise


def consistency_fn(net, x: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """f(x, sigma) = c_skip x + c_out F(x, 1000 sigma, cond); sigma (B,) in [0, sigma_max]."""
    s = (sigma / 0.1).view(-1, 1, 1, 1)
    c_skip = SIGMA_DATA ** 2 / (s ** 2 + SIGMA_DATA ** 2)
    c_out = s / (s ** 2 + SIGMA_DATA ** 2).sqrt()
    return c_skip * x + c_out * net(x, 1000 * sigma, cond).float()


def charbonnier(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-sample mean of sqrt((a - b)^2 + eps): (B,)."""
    return ((a - b) ** 2 + eps).sqrt().mean(dim=(1, 2, 3))


def lpips_gray(lpips_net, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample LPIPS of grey images in [-1, 1], copied to 3 channels: (B,)."""
    return lpips_net(a.expand(-1, 3, -1, -1), b.expand(-1, 3, -1, -1)).view(-1)


def distance(lpips_net, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """d of Eq. 20: 0.5 LPIPS + 0.5 Charbonnier, per sample (B,)."""
    return 0.5 * lpips_gray(lpips_net, a, b) + 0.5 * charbonnier(a, b)


def ct_step(net, lpips_net, x0, y0, cond, intervals: int, sigma_max: float, amp: bool):
    """One CT loss evaluation (Eq. 10). Returns (loss, online output x_hat0) for DTM to reuse."""
    sigmas, alphas = grid(intervals, sigma_max, x0.device)
    index = int(torch.randint(0, intervals, (1,)))                 # one t for the batch, as the code
    noise = torch.randn_like(x0)
    x_cur = q_sample(x0, y0, sigmas[index], alphas[index], noise)             # x_{t-1}
    x_next = q_sample(x0, y0, sigmas[index + 1], alphas[index + 1], noise)    # x_t, same noise
    batch = lambda i: sigmas[i].expand(len(x0))
    with torch.autocast(x0.device.type, dtype=torch.float16, enabled=amp):
        with torch.no_grad():
            target = consistency_fn(net, x_cur, batch(index), cond)
        online = consistency_fn(net, x_next, batch(index + 1), cond)
    return distance(lpips_net, online.float(), target.float()).mean(), online.float()


def dtm_loss(frozen, lpips_net, x_hat0, x0, y0, cond, sigma: float, alpha: float, weight: float, amp: bool):
    """DTM (supp. Alg. 3, 'improved' weight): match the frozen model's trajectories from the noised
    output and the noised ground truth. Gradient reaches x_hat0 only through LPIPS."""
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, y0, sigma, alpha, noise)
    z_t = q_sample(x_hat0.detach(), y0, sigma, alpha, noise)
    s = torch.full((len(x0),), sigma, device=x0.device)
    with torch.no_grad(), torch.autocast(x0.device.type, dtype=torch.float16, enabled=amp):
        grad = consistency_fn(frozen, z_t, s, cond).float() - consistency_fn(frozen, x_t, s, cond).float()
    w = 1 / (x_hat0.detach() - x0).abs().mean(dim=(1, 2, 3)).clamp(min=1e-6)
    return (0.5 * weight * w * lpips_gray(lpips_net, x_hat0, x_hat0.detach() - grad)).mean()


@torch.no_grad()
def one_step(net, y0: torch.Tensor, cond: torch.Tensor, sigma_max: float) -> torch.Tensor:
    """Inference: x_T = y_0 + sigma_max eps (alpha = 1), one network call. Clamped to [-1, 1]."""
    x_t = y0 + sigma_max * torch.randn_like(y0)
    return consistency_fn(net, x_t, torch.full((len(y0),), sigma_max, device=y0.device), cond).clamp(-1, 1)
