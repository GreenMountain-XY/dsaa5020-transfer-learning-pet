"""Model builders for ResNet-18 (primary) and optional lightweight backbones.

Supports four training strategies:
  - scratch    : random initialization, all layers trainable
  - frozen     : ImageNet-1K pretrained, backbone frozen, only classifier head trainable
  - partial_ft : ImageNet-1K pretrained, last stage + classifier head trainable (extension)
  - full_ft    : ImageNet-1K pretrained, all layers trainable
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_BACKBONES = ("resnet18", "mobilenet_v3_small", "efficientnet_b0")
SUPPORTED_STRATEGIES = ("scratch", "frozen", "partial_ft", "full_ft")

DEFAULT_LRS: Dict[str, Dict[str, float]] = {
    "scratch": {"backbone": 1e-3, "head": 1e-3},
    "frozen": {"backbone": 0.0, "head": 3e-3},
    "partial_ft": {"backbone": 3e-4, "head": 1e-3},
    "full_ft": {"backbone": 1e-4, "head": 1e-3},
}

DEFAULT_EPOCHS: Dict[str, Tuple[int, int]] = {
    # (max_epochs, early_stopping_patience)
    "scratch": (80, 20),
    "frozen": (40, 12),
    "partial_ft": (40, 12),
    "full_ft": (40, 12),
}


def _get_head_and_last_stage(
    model: nn.Module, backbone: str
) -> Tuple[nn.Module, nn.Module]:
    """Return (head_module, last_feature_stage_module) for a supported backbone."""
    if backbone == "resnet18":
        return model.fc, model.layer4
    if backbone == "mobilenet_v3_small":
        return model.classifier, model.features[-3:]
    if backbone == "efficientnet_b0":
        return model.classifier, model.features[-2:]
    raise ValueError(f"Unsupported backbone: {backbone}")


def build_model(
    backbone: str = "resnet18",
    strategy: str = "frozen",
    num_classes: int = 37,
    backbone_lr: float | None = None,
    head_lr: float | None = None,
    uniform_lr: bool = False,
) -> Tuple[nn.Module, List[Dict]]:
    """Construct backbone, apply freeze/unfreeze policy, and return (model, param_groups)."""
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"backbone must be one of {SUPPORTED_BACKBONES}, got {backbone}")
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"strategy must be one of {SUPPORTED_STRATEGIES}, got {strategy}")

    pretrained = strategy != "scratch"

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(backbone)

    head_mod, last_stage_mod = _get_head_and_last_stage(model, backbone)

    # Apply parameter freezing policy
    if strategy == "frozen":
        for p in model.parameters():
            p.requires_grad = False
        for p in head_mod.parameters():
            p.requires_grad = True
    elif strategy == "partial_ft":
        for p in model.parameters():
            p.requires_grad = False
        for p in last_stage_mod.parameters():
            p.requires_grad = True
        for p in head_mod.parameters():
            p.requires_grad = True
    else:
        # scratch or full_ft: all parameters trainable
        for p in model.parameters():
            p.requires_grad = True

    # Build optimizer parameter groups
    b_lr = backbone_lr if backbone_lr is not None else DEFAULT_LRS[strategy]["backbone"]
    h_lr = head_lr if head_lr is not None else DEFAULT_LRS[strategy]["head"]
    if uniform_lr:
        h_lr = b_lr if b_lr > 0 else h_lr

    head_param_ids = {id(p) for p in head_mod.parameters()}
    backbone_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids
    ]
    head_params = [
        p for p in model.parameters() if p.requires_grad and id(p) in head_param_ids
    ]

    param_groups: List[Dict] = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": b_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": h_lr})

    return model, param_groups


def set_module_train_mode(model: nn.Module, backbone: str, strategy: str) -> None:
    """Keep frozen BatchNorm layers in eval() mode to preserve pretrained statistics."""
    if strategy in ("scratch", "full_ft"):
        model.train()
        return

    model.eval()
    head_mod, last_stage_mod = _get_head_and_last_stage(model, backbone)
    head_mod.train()
    if strategy == "partial_ft":
        last_stage_mod.train()


def get_classifier_layer(model: nn.Module, backbone: str) -> nn.Module:
    """Return the final linear layer whose input is the penultimate representation."""
    if backbone == "resnet18":
        return model.fc
    if backbone in ("mobilenet_v3_small", "efficientnet_b0"):
        return model.classifier[-1]
    raise ValueError(f"Unsupported backbone: {backbone}")
