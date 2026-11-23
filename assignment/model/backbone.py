"""Model backbone: pretrained ResNet-18 transfer learning.

Default backbone is torchvision ResNet-18 with ImageNet-1K weights, chosen
because it is lightweight enough to train on an RTX 3050 6GB while still
benaching from strong pretrained features. The final fully-connected layer is
replaced with a new head for the five target categories.

Nothing here touches the existing webcam application or its BioCLIP/ConvNeXt
components.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 5
    freeze_backbone: bool = False          # True = feature extraction, False = fine-tune all
    dropout: float = 0.0
    device: str = "cuda"                   # "cuda" | "cpu" | "auto"


def build_model(config: ModelConfig | None = None) -> tuple[nn.Module, str]:
    """Build a ResNet-18 classifier with a replaced classification head.

    Returns ``(model, device)``. The model is moved to ``device`` and set to
    train mode by default.
    """
    config = config or ModelConfig()
    device = _resolve_device(config.device)

    weights = ResNet18_Weights.IMAGENET1K_V1 if config.pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    head_layers: list[nn.Module] = []
    if config.dropout and config.dropout > 0:
        head_layers.append(nn.Dropout(p=config.dropout))
    head_layers.append(nn.Linear(in_features, config.num_classes))
    model.fc = nn.Sequential(*head_layers)

    if config.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    model = model.to(device)
    return model, device


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; use device='cpu' or 'auto'")
    return requested


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    model: nn.Module,
    config: ModelConfig,
    path: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": {
            "backbone": config.backbone,
            "pretrained": config.pretrained,
            "num_classes": config.num_classes,
            "freeze_backbone": config.freeze_backbone,
            "dropout": config.dropout,
            "device": config.device,
        },
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[nn.Module, ModelConfig, dict[str, Any]]:
    """Load a checkpoint and return (model, config, extra)."""
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = payload.get("config", {})
    config = ModelConfig(**cfg_dict)
    model, _ = build_model(config)
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(_resolve_device(device))
    model.eval()
    return model, config, payload.get("extra", {})
