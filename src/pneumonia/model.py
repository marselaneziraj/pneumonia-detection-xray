"""Backbones with a single-logit head.

Each model exposes `.features` (the last convolutional feature map, the Grad-CAM target),
`.pool` and `.head`. The forward is written out explicitly (no in-place ReLU after the
feature map) so Grad-CAM hooks on `.features` see clean activations and gradients.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED = ("densenet121", "resnet18")


class PneumoniaNet(nn.Module):
    def __init__(self, features: nn.Module, n_features: int, dropout: float, relu_after: bool):
        super().__init__()
        self.features = features
        self.relu_after = relu_after
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        if self.relu_after:  # DenseNet's features end at norm5; torchvision applies ReLU after
            f = F.relu(f)
        return self.head(torch.flatten(self.pool(f), 1)).squeeze(1)


def build_model(arch: str, pretrained: bool = True, dropout: float = 0.0) -> PneumoniaNet:
    from torchvision import models

    arch = arch.lower()
    if arch == "densenet121":
        w = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = _load(models.densenet121, w)
        return PneumoniaNet(net.features, net.classifier.in_features, dropout, relu_after=True)
    if arch == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = _load(models.resnet18, w)
        feats = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        return PneumoniaNet(feats, net.fc.in_features, dropout, relu_after=False)
    raise ValueError(f"Unknown arch {arch!r}; supported: {SUPPORTED}")


def _load(fn, weights):
    try:
        return fn(weights=weights)
    except Exception as e:  # network failure while fetching ImageNet weights
        raise RuntimeError(
            f"Could not load ImageNet weights {weights} ({e}). Check internet access to "
            "download.pytorch.org, or set model.pretrained: false (from-scratch baseline)."
        ) from e


def set_backbone_trainable(model: PneumoniaNet, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def init_head_bias(model: PneumoniaNet, prior_pos: float) -> None:
    """Start the logit at the log-odds of the training prior (stabilises phase 1)."""
    import math

    prior_pos = min(max(prior_pos, 1e-4), 1 - 1e-4)
    with torch.no_grad():
        model.head[-1].bias.fill_(math.log(prior_pos / (1 - prior_pos)))
