"""Training loop for the five-class fish classifier.

Features:
  - Transfer learning from pretrained ResNet-18.
  - Configurable batch size, learning rate, epochs, optimizer (SGD/Adam),
    weight decay, dropout.
  - CUDA when available; CPU fallback.
  - Early stopping on validation loss.
  - Best-validation checkpoint saved.
  - Train/val loss and accuracy recorded per epoch.
  - Full training configuration and random seed saved.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .model.backbone import ModelConfig, build_model, count_trainable_parameters, count_total_parameters
from .data.dataset import FishCategoryDataset, build_datasets, eval_transform, train_transform
from .data.split import SplitResult, split_entries, save_split
from .data.manifest import load_manifest, manifest_summary, save_manifest_json, validate_manifest


@dataclass
class TrainConfig:
    seed: int = 1337
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 30
    optimizer: str = "adamw"           # "adamw" | "adam" | "sgd"
    weight_decay: float = 1e-4
    dropout: float = 0.0
    freeze_backbone: bool = False
    device: str = "auto"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    num_workers: int = 2
    pin_memory: bool = True
    # Data gating
    verified_human_statuses: tuple[str, ...] = ()
    treat_needs_review_as_ineligible: bool = True
    # Paths
    run_dir: Path | None = None
    image_root: Path | None = None


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic behaviour where possible (may reduce speed slightly).
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        predicted = logits.argmax(dim=1)
        correct = (predicted == targets).sum().item()
        return correct / max(1, targets.size(0))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_acc += _accuracy(logits, targets)
        n_batches += 1
    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += float(loss.item())
        total_acc += _accuracy(logits, targets)
        n_batches += 1
    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


def _make_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    trainable = [p for p in model.parameters() if p.requires_grad]
    name = config.optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(
            trainable, lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError(f"Unknown optimizer: {config.optimizer!r}")


def run_training(
    manifest_entries,
    categories,
    config: TrainConfig | None = None,
) -> dict[str, Any]:
    """Prepare data, train, and return a results dictionary (no I/O side
    effects beyond writing to ``config.run_dir`` when provided)."""
    config = config or TrainConfig()
    set_seed(config.seed)

    device = "cuda" if (config.device == "auto" and torch.cuda.is_available()) else config.device
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    # --- Data preparation (deterministic, group-aware, stratified) ---
    split_result = split_entries(manifest_entries, seed=config.seed)
    datasets = build_datasets(
        split_result,
        categories,
        image_root=config.image_root,
    )
    # Apply transforms per split (build_datasets already sets them).
    loaders = {
        s: DataLoader(
            datasets[s],
            batch_size=config.batch_size,
            shuffle=(s == "train"),
            num_workers=config.num_workers,
            pin_memory=config.pin_memory and device == "cuda",
            generator=torch.Generator().manual_seed(config.seed),
        )
        for s in ("train", "val", "test")
    }
    # Only iterate test loader at final evaluation; never during training.

    # --- Model ---
    model_config = ModelConfig(
        num_classes=len(categories),
        freeze_backbone=config.freeze_backbone,
        dropout=config.dropout,
        device=device,
    )
    model, device = build_model(model_config)

    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, config)

    # --- Training loop with early stopping ---
    history: list[EpochMetrics] = []
    best_val_loss = math.inf
    best_epoch = -1
    patience_counter = 0
    best_state = None

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate_split(model, loaders["val"], criterion, device)
        history.append(
            EpochMetrics(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                         val_loss=val_loss, val_acc=val_acc)
        )

        improved = val_loss < best_val_loss - config.early_stopping_min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            break

    # Restore best checkpoint.
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Final evaluation on held-out test set ---
    test_loss, test_acc = evaluate_split(model, loaders["test"], criterion, device)

    results = {
        "seed": config.seed,
        "device": device,
        "config": asdict(config),
        "model_config": asdict(model_config),
        "total_parameters": count_total_parameters(model),
        "trainable_parameters": count_trainable_parameters(model),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "history": [asdict(h) for h in history],
        "split_summary": split_result.summary(),
        "data_summary": manifest_summary(manifest_entries),
        "stopped_early": patience_counter >= config.early_stopping_patience,
        "epochs_run": len(history),
    }
    return results, model, split_result, datasets


def save_run(results: dict[str, Any], model: nn.Module, split_result: SplitResult,
             categories, run_dir: Path) -> None:
    """Persist a completed run: config, history, split manifest, checkpoint."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Training configuration + seed.
    (run_dir / "training_config.json").write_text(
        json.dumps(results["config"], indent=2), encoding="utf-8"
    )
    # History.
    (run_dir / "history.json").write_text(
        json.dumps(results["history"], indent=2), encoding="utf-8"
    )
    # Split manifest (train/val/test membership).
    save_split(split_result, run_dir / "split_manifest.json")
    # Best checkpoint.
    from .model.backbone import save_checkpoint, ModelConfig
    mc = ModelConfig(**results["model_config"])
    save_checkpoint(model, mc, run_dir / "best_model.pt", extra={"results_summary": {
        "test_accuracy": results["test_accuracy"],
        "test_loss": results["test_loss"],
        "best_epoch": results["best_epoch"],
    }})
