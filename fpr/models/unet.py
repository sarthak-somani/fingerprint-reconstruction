"""Small conditional U-Net (ResShift denoiser, all ResShift experiments): predicts the clean print x_0 from x_t, t and cond.

Input: x_t (B, 1, H, W) and cond (B, C, H, W) concatenated on channels; t (B,) long.
Output: x0_hat (B, 1, H, W) in roughly [-1, 1]. H and W must be divisible by 2^(levels-1).
Sizes come from the experiment's params.json ('model'); the default is about 8M parameters.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    angles = t.float()[:, None] * freqs[None]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> conv, plus the time embedding, twice; 1x1 skip if channels change."""

    def __init__(self, c_in: int, c_out: int, t_dim: int, groups: int):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(groups, c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.time = nn.Linear(t_dim, c_out)
        self.norm2, self.conv2 = nn.GroupNorm(groups, c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Single-head self-attention over all pixels of a (B, C, H, W) feature map."""

    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv, self.proj = nn.Conv2d(channels, 3 * channels, 1), nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return x + self.proj(out.transpose(1, 2).reshape(b, c, h, w))


class UNet(nn.Module):
    """Encoder-decoder with skip connections; attention at the listed resolutions (patch 128)."""

    def __init__(self, cond_channels: int = 4, base_channels: int = 32, channel_mults=(1, 2, 4, 5),
                 res_blocks: int = 2, attention_resolutions=(16,), groups: int = 8, patch: int = 128, **_):
        super().__init__()
        t_dim = 4 * base_channels
        self.base_channels = base_channels
        self.time_mlp = nn.Sequential(nn.Linear(base_channels, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.conv_in = nn.Conv2d(1 + cond_channels, base_channels, 3, padding=1)

        self.down, skips, c, size = nn.ModuleList(), [base_channels], base_channels, patch
        for level, mult in enumerate(channel_mults):
            for _ in range(res_blocks):
                self.down.append(ResBlock(c, base_channels * mult, t_dim, groups))
                c = base_channels * mult
                if size in attention_resolutions:
                    self.down.append(AttentionBlock(c, groups))
                skips.append(c)
            if level < len(channel_mults) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                skips.append(c)
                size //= 2

        self.middle = nn.ModuleList([ResBlock(c, c, t_dim, groups), AttentionBlock(c, groups),
                                     ResBlock(c, c, t_dim, groups)])

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            for _ in range(res_blocks + 1):
                self.up.append(ResBlock(c + skips.pop(), base_channels * mult, t_dim, groups))
                c = base_channels * mult
                if size in attention_resolutions:
                    self.up.append(AttentionBlock(c, groups))
            if level > 0:
                self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))
                self.up.append(nn.Conv2d(c, c, 3, padding=1))
                size *= 2
        self.out = nn.Sequential(nn.GroupNorm(groups, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(t, self.base_channels))
        h = self.conv_in(torch.cat([x_t, cond], dim=1))
        skips = [h]
        for layer in self.down:
            h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
            if not isinstance(layer, AttentionBlock):
                skips.append(h)
        for layer in self.middle:
            h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
        for layer in self.up:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, skips.pop()], dim=1), t_emb)
            else:
                h = layer(h)
        return self.out(h)
