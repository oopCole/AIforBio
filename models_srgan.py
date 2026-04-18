from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn2(self.conv2(self.prelu(self.bn1(self.conv1(x)))))
        return x + y


class SRGANGenerator(nn.Module):
    """32x32 -> 128x128 (4x), two 2x upsampling steps via sub-pixel conv."""

    def __init__(self, n_residual_blocks: int = 8) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, 9, padding=4, bias=False)
        self.prelu = nn.PReLU()
        blocks: list[nn.Module] = [ResidualBlock(64) for _ in range(n_residual_blocks)]
        self.res = nn.Sequential(*blocks)
        self.conv_mid = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn_mid = nn.BatchNorm2d(64)
        # upsample x2 twice: 32->64->128
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
        self.conv_out = nn.Conv2d(64, 3, 9, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x in [-1, 1]
        f0 = self.prelu(self.conv_in(x))
        f = self.res(f0)
        f = self.bn_mid(self.conv_mid(f)) + f0
        f = self.up1(f)
        f = self.up2(f)
        return torch.tanh(self.conv_out(f))


class SRGANDiscriminator(nn.Module):
    """PatchGAN-style discriminator for 128x128."""

    def __init__(self) -> None:
        super().__init__()
        def block(cin: int, cout: int, stride: int, norm: bool = True) -> nn.Sequential:
            layers: list[nn.Module] = [nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(3, 64, 2, norm=False),
            block(64, 128, 2),
            block(128, 256, 2),
            block(256, 512, 2),
            nn.Conv2d(512, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VGGPerceptualLoss(nn.Module):
    """Feature loss using VGG19 relu activations before conv5 (approx. paper)."""

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        # freeze
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # sr, hr in [0,1] -> ImageNet norm for VGG
        def norm(t: torch.Tensor) -> torch.Tensor:
            return (t - self.mean) / self.std

        f_sr = self.vgg(norm(sr))
        f_hr = self.vgg(norm(hr))
        return nn.functional.l1_loss(f_sr, f_hr)


def weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
