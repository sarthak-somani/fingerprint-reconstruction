"""FingerGAN generator (Zhu, Yin & Hu, TPAMI 2023), copied from HubYZ/LatentEnhancement model/my_network.py.

Only the generator is kept (the discriminator is not needed to run the released weights).
Layer and attribute names are unchanged, including `out_blcok`, so the released state dict loads.
U-Net with valid 3x3 convolutions, BatchNorm and ELU, 64 -> 1024 channels, sigmoid output; 33.9M params.
"""
import torch
import torch.nn as nn


class InitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3), nn.BatchNorm2d(out_channels), nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3), nn.BatchNorm2d(out_channels), nn.ELU())

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2), nn.BatchNorm2d(out_channels), nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3), nn.BatchNorm2d(out_channels), nn.ELU())

    def forward(self, x):
        return self.double_conv(x)


class BottleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2), nn.BatchNorm2d(out_channels), nn.ELU())
        self.bottle_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3), nn.BatchNorm2d(out_channels), nn.ELU())
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3), nn.BatchNorm2d(out_channels), nn.ELU())

    def forward(self, x):
        return self.up_conv(self.bottle_conv(self.down_conv(x)))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels), nn.ELU())
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3), nn.BatchNorm2d(out_channels), nn.ELU())

    def forward(self, x_current, x_previous):
        return self.up_conv2(torch.cat([x_previous, self.up_conv1(x_current)], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))


class FingerGAN(nn.Module):
    """(B, 1, 192, 192) texture image in [0, 1] -> (B, 1, 192, 192) enhanced print in [0, 1]."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.init_block = InitBlock(in_channels, 64)
        self.down1, self.down2, self.down3 = DownBlock(64, 128), DownBlock(128, 256), DownBlock(256, 512)
        self.bottle = BottleBlock(512, 1024)
        self.up1, self.up2 = UpBlock(1024, 512), UpBlock(512, 256)
        self.up3, self.up4 = UpBlock(256, 128), UpBlock(128, 64)
        self.out_blcok = OutConv(64, in_channels)                      # sic: name of the released weights

    def forward(self, x):
        x1 = self.init_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.bottle(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_blcok(x)
