"""Tests for E02_resshift: the residual-shift schedule, forward process, loss and sampler."""
import pytest
import torch

from fpr.models import resshift
from fpr.models.schedule import build_schedule, DIFFUSION, respace, sqrt_eta_schedule


def toy_batch(b: int = 4, size: int = 32):
    gen = torch.Generator().manual_seed(0)
    x0 = torch.rand(b, 1, size, size, generator=gen) * 2 - 1
    mask = torch.zeros(b, 1, size, size)
    mask[..., 8:24, 8:24] = 1
    y0 = x0 * (1 - mask) + 0.7 * mask                    # damage = flat grey patch
    return x0, y0, mask


class TinyModel(torch.nn.Module):
    """Stand-in for the team's network: one conv on [x_t, cond], ignores t."""
    def __init__(self, cond_channels: int = 2):
        super().__init__()
        self.conv = torch.nn.Conv2d(1 + cond_channels, 1, 3, padding=1)

    def forward(self, x_t, t, cond):
        return self.conv(torch.cat([x_t, cond], dim=1))


def test_schedule_endpoints_and_monotone():
    sqrt_eta = sqrt_eta_schedule()
    assert len(sqrt_eta) == DIFFUSION["steps"]
    assert sqrt_eta[0] == pytest.approx(DIFFUSION["min_noise"] / DIFFUSION["kappa"])
    assert sqrt_eta[-1] == pytest.approx(DIFFUSION["sqrt_eta_end"])
    assert (torch.tensor(sqrt_eta).diff() > 0).all()
    s = build_schedule()
    assert s["eta"][0] == 0 and s["post_var"][1] == 0 and s["coef_x0"][1] == pytest.approx(1.0)


def test_forward_centred_on_x0_at_t0_and_on_y0_at_T():
    x0, y0, _ = toy_batch()
    s = build_schedule()
    zero = torch.zeros(len(x0), dtype=torch.long)
    big_t = torch.full_like(zero, s["steps"])
    no_noise = torch.zeros_like(x0)
    assert torch.equal(resshift.q_sample(x0, y0, zero, no_noise, s), x0)
    mean_t = resshift.q_sample(x0, y0, big_t, no_noise, s)
    # eta_T = 0.99**2, so the mean is y0 up to (1 - eta_T) of the residual.
    assert (mean_t - y0).abs().max() <= (1 - s["eta"][-1]) * (y0 - x0).abs().max() + 1e-6
    samples = torch.stack([resshift.q_sample(x0, y0, big_t, torch.randn_like(x0), s) for _ in range(400)])
    assert (samples.mean(0) - mean_t).abs().mean() < 0.1
    assert samples.std(0).mean() == pytest.approx(s["kappa"] * s["eta"][-1].sqrt(), rel=0.1)


def test_loss_finite_and_differentiable():
    x0, y0, mask = toy_batch()
    model = TinyModel()
    loss = resshift.training_loss(model, x0, y0, mask, torch.cat([y0, mask], dim=1))
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("steps", [None, 4])
def test_sample_shape_range_and_paste(steps):
    x0, y0, mask = toy_batch()
    out = resshift.sample(TinyModel(), y0, mask, torch.cat([y0, mask], dim=1), steps=steps)
    assert out.shape == y0.shape and out.min() >= -1 and out.max() <= 1
    assert torch.equal(out[mask == 0], y0[mask == 0])


def test_perfect_model_recovers_x0():
    """A model that always returns x_0 makes the sampler end exactly on x_0 (Eq. (7)-(9))."""
    x0, y0, mask = toy_batch()
    out = resshift.sample(lambda x_t, t, cond: x0, y0, mask, None)
    assert torch.allclose(out, x0, atol=1e-6)


def test_respace_keeps_first_step_and_model_timesteps():
    s = respace(build_schedule(), 4)
    assert s["steps"] == 4 and s["model_t"].tolist() == [0, 1, 4, 8, 12]
    assert s["post_var"][1] == 0
