"""Tests for the ResNet-18 backbone (CPU, no training)."""

from __future__ import annotations

from pathlib import Path

import torch

from assignment.model.backbone import (
    ModelConfig,
    build_model,
    count_trainable_parameters,
    count_total_parameters,
    load_checkpoint,
    save_checkpoint,
)


def test_build_model_forward_shape_cpu():
    model, device = build_model(ModelConfig(num_classes=5, device="cpu"))
    assert device == "cpu"
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 5)


def test_parameter_counts():
    model, _ = build_model(ModelConfig(num_classes=5, device="cpu"))
    total = count_total_parameters(model)
    trainable = count_trainable_parameters(model)
    assert total == trainable  # nothing frozen by default
    assert total > 10_000_000  # ResNet-18 is ~11M


def test_freeze_backbone_freezes_all_but_head():
    model, _ = build_model(ModelConfig(num_classes=5, freeze_backbone=True, device="cpu"))
    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)
    assert 0 < trainable < total
    # The head (fc) should be trainable; backbone frozen.
    head_trainable = sum(p.numel() for p in model.fc.parameters() if p.requires_grad)
    assert head_trainable > 0


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    model, _ = build_model(ModelConfig(num_classes=5, device="cpu"))
    ckpt = tmp_path / "model.pt"
    save_checkpoint(model, ModelConfig(num_classes=5, device="cpu"), ckpt, extra={"note": "test"})
    loaded, config, extra = load_checkpoint(ckpt, device="cpu")
    assert config.num_classes == 5
    assert extra["note"] == "test"
    loaded.eval()
    with torch.no_grad():
        a = model(torch.randn(1, 3, 224, 224))
        b = loaded(torch.randn(1, 3, 224, 224))
    assert a.shape == b.shape == (1, 5)
