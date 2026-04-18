from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet18_binary(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, num_classes)
    return m
