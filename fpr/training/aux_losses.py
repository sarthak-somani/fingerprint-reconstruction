"""Auxiliary losses on ResShift's predicted clean print x0_hat (E06-E08, E12-E16, E26; final model E22 = E07).

Each term compares the composite xc (x0_hat inside the dilated damage mask M, the clean x0 outside)
with x0, only inside M, and returns one value per sample. aux_loss applies the terms at low-noise
steps only (eta_t <= eta_max) and sums them. Tensors are (B, 1, H, W); images in [0, 1] unless named
otherwise. Terms: helpers + minutiae_term (E06), cldice_term (E07),
cv_edge_term + cv_mix_term (E08).
"""
import torch
import torch.nn.functional as F

EPS = 1e-6


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] as (x + 1) / 2. No clamp, so gradients survive outside the range."""
    return (x + 1) / 2


def dilate(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Binary mask (B, 1, H, W) grown by px pixels in every direction (a square max filter)."""
    return F.max_pool2d(mask, kernel_size=2 * px + 1, stride=1, padding=px)


def composite(x0_hat01: torch.Tensor, x0_01: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """m * x0_hat + (1 - m) * x0: the prediction inside the mask, the clean print outside.

    x0 and m are detached, so gradients flow only through x0_hat.
    """
    m = m.detach()
    return m * x0_hat01 + (1 - m) * x0_01.detach()


def masked_mean(values: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Per-sample mean of values (B, 1, H, W) over the pixels where m = 1 -> (B,). 0 if m is empty."""
    return (values * m).sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2, 3)).clamp_min(EPS)


def minutiae_term(xc: torch.Tensor, x0: torch.Tensor, m: torch.Tensor, w_norm: torch.Tensor,
                  lam_m: float) -> torch.Tensor:
    """Minutiae-weighted L1 -> (B,): sum_M(w |xc - x0|) / sum(M), with w = 1 + lam_m * w_norm.

    w_norm (B, 1, H, W) in [0, 1]: Gaussian bumps (sigma 8 px) at the clean print's MINDTCT
    minutiae (preprocessing/09_make_minutiae_maps.py). Pixels near a minutia count up
    to 1 + lam_m times as much as plain ridge pixels.
    """
    w = 1 + lam_m * w_norm.detach()
    return masked_mean(w * (xc - x0).abs(), m)


def cldice_term(xc: torch.Tensor, x0: torch.Tensor, m: torch.Tensor, tau: float = 0.05,
                iters: int = 4) -> torch.Tensor:
    """1 - clDice (Shit et al., CVPR 2021) between the soft ridge maps of xc and x0, inside M -> (B,).

    Spec (agent 2, E07):
    - soft ridge map r(x) = sigmoid((local_mean_11x11(x) - x) / tau), local mean =
      avg_pool2d(11, stride 1, padding 5, count_include_pad=False); ~1 on dark ridges
      (checked on real clean crops: ~0.96 on Sauvola ridges), 0.5 on flat background;
    - V_P = r(xc), V_L = r(x0); S_P, S_L = fpr.training.skeleton.soft_skeleton(V, iterations=iters);
    - per sample, sums over pixels, eps = 1e-6:
      Tprec = (sum S_P V_L M + eps) / (sum S_P M + eps), Tsens = (sum S_L V_P M + eps) / (sum S_L M + eps),
      loss = 1 - 2 Tprec Tsens / (Tprec + Tsens).
    """
    from fpr.training.skeleton import soft_skeleton         # imported here: only the clDice term needs it

    def ridges(x: torch.Tensor) -> torch.Tensor:
        local_mean = F.avg_pool2d(x, 11, stride=1, padding=5, count_include_pad=False)
        return torch.sigmoid((local_mean - x) / tau)

    def sum_in_mask(values: torch.Tensor) -> torch.Tensor:
        return (values * m.detach()).sum(dim=(1, 2, 3))

    v_p, v_l = ridges(xc), ridges(x0.detach())
    s_p, s_l = soft_skeleton(v_p, iterations=iters), soft_skeleton(v_l, iterations=iters)
    t_prec = (sum_in_mask(s_p * v_l) + EPS) / (sum_in_mask(s_p) + EPS)    # pred skeleton on clean ridges
    t_sens = (sum_in_mask(s_l * v_p) + EPS) / (sum_in_mask(s_l) + EPS)    # clean skeleton on pred ridges
    return 1 - 2 * t_prec * t_sens / (t_prec + t_sens)


_LAPLACIAN = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]   # 4-neighbour Laplacian (E08)
_SSIM_WINDOW, _SSIM_SIGMA = 11, 1.5                                 # Wang et al. 2004's Gaussian window
_SSIM_C1, _SSIM_C2 = 0.01 ** 2, 0.03 ** 2                           # (K * data_range)^2, data_range 1
_MS_SSIM_WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363]                 # Wang et al. 2003's first 4 scales


def _laplacian(x: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) -> (B, 1, H, W): the fixed 3x3 Laplacian, zero padding 1."""
    kernel = torch.tensor(_LAPLACIAN, dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    return F.conv2d(x, kernel, padding=1)


def _ssim_maps(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(ssim, cs) maps of (B, 1, H, W) images in [0, 1], each (B, 1, H - 10, W - 10).

    Gaussian window 11, sigma 1.5, valid convolution (no padding); cs is the contrast-structure
    part, ssim = luminance * cs. Population (not sample) variances, as skimage's
    use_sample_covariance=False.
    """
    taps = torch.arange(_SSIM_WINDOW, dtype=x.dtype, device=x.device) - _SSIM_WINDOW // 2
    g = torch.exp(-taps ** 2 / (2 * _SSIM_SIGMA ** 2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).view(1, 1, _SSIM_WINDOW, _SSIM_WINDOW)
    mu_x, mu_y = F.conv2d(x, window), F.conv2d(y, window)
    var_x = F.conv2d(x * x, window) - mu_x ** 2
    var_y = F.conv2d(y * y, window) - mu_y ** 2
    cov = F.conv2d(x * y, window) - mu_x * mu_y
    cs = (2 * cov + _SSIM_C2) / (var_x + var_y + _SSIM_C2)
    luminance = (2 * mu_x * mu_y + _SSIM_C1) / (mu_x ** 2 + mu_y ** 2 + _SSIM_C1)
    return luminance * cs, cs


def _ms_ssim_in_mask(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """MS-SSIM of (B, 1, H, W) images in [0, 1], each scale's map averaged over M -> (B,).

    4 scales, 2x avg-pool between them (images and M alike); M is cropped by 5 px per side to
    match the valid maps. Scales 1-3 give cs, scale 4 gives ssim; MS-SSIM = prod value_j ** w_j.
    A scale where the cropped M is empty (damage only near the patch border, ~4% of gated train
    patches at scale 4) counts as 1, i.e. drops out, instead of zeroing the whole product.
    """
    weights = torch.tensor(_MS_SSIM_WEIGHTS, dtype=x.dtype, device=x.device)
    weights, crop, values = weights / weights.sum(), _SSIM_WINDOW // 2, []
    for scale in range(len(_MS_SSIM_WEIGHTS)):
        if scale > 0:
            x, y, m = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2), F.avg_pool2d(m, 2)
        ssim, cs = _ssim_maps(x, y)
        m_valid = m[..., crop:-crop, crop:-crop]
        value = masked_mean(ssim if scale == len(_MS_SSIM_WEIGHTS) - 1 else cs, m_valid)
        value = torch.where(m_valid.sum(dim=(1, 2, 3)) > 0, value, torch.ones_like(value))
        values.append(value.relu())                          # relu before the power: no NaN
    return torch.prod(torch.stack(values, dim=1) ** weights, dim=1)


def cv_edge_term(xc: torch.Tensor, x0: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Charbonnier distance between Laplacian edge maps, inside M -> (B,).

    masked_mean over M of sqrt((Lap(xc) - Lap(x0))^2 + 1e-6), with Lap the fixed 3x3 Laplacian
    [[0, 1, 0], [1, -4, 1], [0, 1, 0]] (conv2d, padding 1). Lap picks out ridge edges, so the term
    rewards sharp ridge walls. Its floor is sqrt(1e-6) = 1e-3 when xc = x0.
    """
    diff = _laplacian(xc) - _laplacian(x0.detach())
    return masked_mean(torch.sqrt(diff ** 2 + EPS), m.detach())


def cv_mix_term(xc: torch.Tensor, x0: torch.Tensor, m: torch.Tensor, alpha: float = 0.84) -> torch.Tensor:
    """MS-SSIM / L1 mix (Zhao et al., IEEE TCI 2017, Eq. 14), inside M -> (B,).

    alpha (1 - MS_SSIM_M(xc, x0)) + (1 - alpha) L1_M, with MS_SSIM_M from _ms_ssim_in_mask
    (Wang et al. 2003: window 11, sigma 1.5, C1 = 0.01^2, C2 = 0.03^2, 4 scales 128 -> 16 px,
    Wang's first 4 scale weights renormalised) and L1_M = masked_mean |xc - x0|.
    alpha = 0.84 is Zhao et al.'s value. Unlike them, (a) the L1 part is a plain masked L1, not
    Gaussian-weighted; (b) MS-SSIM is Wang's (downsampled scales, one sigma, Wang's weights),
    not theirs (5 Gaussians sigma 0.5-8 on the full-resolution patch, all exponents 1, centre pixel).
    """
    x0, m = x0.detach(), m.detach()
    l1 = masked_mean((xc - x0).abs(), m)
    return alpha * (1 - _ms_ssim_in_mask(xc, x0, m)) + (1 - alpha) * l1


TERMS = {"minutiae": minutiae_term, "cldice": cldice_term, "cv_edge": cv_edge_term, "cv_mix": cv_mix_term}


def eta_gate(eta_t: torch.Tensor, m: torch.Tensor, eta_max: float) -> torch.Tensor:
    """bool (B,): the samples that get the extra terms, where eta_t <= eta_max and the dilated
    mask m is not empty. eta_t: (B,) or (B, 1, 1, 1), the noise level of each sample's step."""
    return (eta_t.reshape(-1) <= eta_max) & (m.sum(dim=(1, 2, 3)) > 0)


def term_lam(setting) -> float:
    """lam of one 'aux_losses' entry of params.json: a number, or {"lam": ..., <arguments>}."""
    return float(setting["lam"] if isinstance(setting, dict) else setting)


def term_kwargs(name: str, setting, aux_cfg: dict, minutiae_w: torch.Tensor | None) -> dict:
    """Keyword arguments of one term: its own from 'aux_losses' plus the shared ones from 'aux'."""
    kwargs = {k: v for k, v in setting.items() if k != "lam"} if isinstance(setting, dict) else {}
    if name == "minutiae":
        assert minutiae_w is not None, "the minutiae term needs the batch's 'minutiae_w' maps"
        kwargs.update(w_norm=minutiae_w, lam_m=aux_cfg["lam_m"])
    if name == "cldice":
        kwargs.update(tau=aux_cfg["cldice_tau"], iters=aux_cfg["cldice_iters"])
    return kwargs


def aux_loss(x0_hat: torch.Tensor, x0: torch.Tensor, mask: torch.Tensor, eta_t: torch.Tensor,
             aux_losses_cfg: dict, aux_cfg: dict, minutiae_w: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, dict]:
    """(total, logs): the weighted sum of the enabled extra terms, and what to log.

    x0_hat, x0: (B, 1, H, W) in [-1, 1] (the model's range); mask: (B, 1, H, W) in {0, 1};
    eta_t: (B,); aux_losses_cfg / aux_cfg: params.json's 'aux_losses' / 'aux'; minutiae_w: the
    batch's (B, 1, H, W) minutiae maps. Each term is averaged over the gated samples (0 if none);
    total = sum of lam * term, a scalar. logs: detached scalars <name> (raw term),
    weighted_<name> (lam * term) and n_gated. Runs in fp32 and uses no random numbers.
    """
    with torch.autocast(x0_hat.device.type, enabled=False):
        x0_hat01, x0_01 = to_unit(x0_hat.float()), to_unit(x0.float()).detach()
        m = dilate(mask.float(), aux_cfg["dilate_px"]).detach()
        xc = composite(x0_hat01, x0_01, m)
        gate = eta_gate(eta_t, m, aux_cfg["eta_max"])
        total = torch.zeros((), device=x0_hat.device)
        logs = {"n_gated": gate.sum().detach()}
        for name, setting in aux_losses_cfg.items():
            lam, kwargs = term_lam(setting), term_kwargs(name, setting, aux_cfg, minutiae_w)
            kwargs = {k: v[gate].float() if torch.is_tensor(v) else v for k, v in kwargs.items()}
            term = TERMS[name](xc[gate], x0_01[gate], m[gate], **kwargs).mean() if gate.any() else total * 0
            total = total + lam * term
            logs[name], logs[f"weighted_{name}"] = term.detach(), (lam * term).detach()
    return total, logs
