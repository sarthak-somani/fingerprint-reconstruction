"""MSU-LatentAFIS enhancement autoencoder, ported from TF/tensorpack to PyTorch.

Source: prip-lab/MSU-LatentAFIS extraction/enhancement_AEC.py, Model.reconstruction2 (MIT licence).
Encoder: five 4x4 stride-2 convs (16, 32, 64, 128, 128 channels, BN + ReLU), a 1x1 conv to 16 channels
and tanh: a 4x4x16 code for a 128 px patch. Decoder: five 4x4 stride-2 deconvs (128, 64, 32, 16, 16,
BN + ReLU), a 3x3 conv to 1 channel and tanh. No skip connections. The LeakyReLUs that follow
BN + ReLU in the original are no-ops on non-negative inputs and are left out.
"""
import torch
import torch.nn as nn


def conv_bn_relu(cin: int, cout: int, kernel: int = 4, stride: int = 2) -> nn.Sequential:
    """tensorpack Conv2D(nl=BNReLU) with 'same' padding."""
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, stride, padding=(kernel - stride + 1) // 2),
                         nn.BatchNorm2d(cout), nn.ReLU())


def deconv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    """tensorpack Deconv2D(kernel 4, stride 2, nl=BNReLU): doubles the size."""
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, padding=1), nn.BatchNorm2d(cout), nn.ReLU())


class EnhancementAEC(nn.Module):
    """(B, 1, H, W) in [-1, 1], H and W multiples of 32 -> (B, 1, H, W) in [-1, 1]."""

    def __init__(self, nf: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_bn_relu(1, nf), conv_bn_relu(nf, nf * 2), conv_bn_relu(nf * 2, nf * 4),
            conv_bn_relu(nf * 4, nf * 8), conv_bn_relu(nf * 8, nf * 8),
            conv_bn_relu(nf * 8, nf, kernel=1, stride=1), nn.Tanh())
        self.decoder = nn.Sequential(
            deconv_bn_relu(nf, nf * 8), deconv_bn_relu(nf * 8, nf * 4), deconv_bn_relu(nf * 4, nf * 2),
            deconv_bn_relu(nf * 2, nf), deconv_bn_relu(nf, nf),
            nn.Conv2d(nf, 1, 3, 1, padding=1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
