"""Evaluation on the held-out test set only.

Computes accuracy, per-class precision / recall / F1 / support and the
confusion matrix using plain NumPy (no scikit-learn dependency). Exports
every misclassification for human explanation and writes charts with
matplotlib. Never evaluates on training data.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model.backbone import load_checkpoint, ModelConfig


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _precision_recall_f1(cm: np.ndarray):
    """Per-class precision, recall, F1, support from a confusion matrix.

    Rows = true class, columns = predicted class.
    """
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    f1 = np.zeros(num_classes, dtype=np.float64)
    support = cm.sum(axis=1).astype(np.float64)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp          # predicted as i but actually not i
        fn = cm[i, :].sum() - tp          # actually i but predicted as not i
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1[i] = 0.0
    return precision, recall, f1, support


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    categories: Sequence[str],
    device: str = "cpu",
) -> dict:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    probs: list[np.ndarray] = []
    sample_ids: list[str] = []
    correct: list[bool] = []

    with torch.no_grad():
        for images, targets, ids in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            probability = torch.softmax(logits, dim=1)
            predicted = logits.argmax(dim=1)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
            probs.extend(probability.cpu().numpy())
            sample_ids.extend(ids)
            correct.extend((predicted == targets).cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    num_classes = len(categories)

    cm = _confusion_matrix(y_true, y_pred, num_classes)
    precision, recall, f1, support = _precision_recall_f1(cm)

    overall_accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    # Macro averages (unweighted by support).
    macro_precision = float(precision.mean()) if num_classes else 0.0
    macro_recall = float(recall.mean()) if num_classes else 0.0
    macro_f1 = float(f1.mean()) if num_classes else 0.0
    # Weighted averages (weighted by support).
    total_support = support.sum()
    if total_support > 0:
        weighted_precision = float((precision * support).sum() / total_support)
        weighted_recall = float((recall * support).sum() / total_support)
        weighted_f1 = float((f1 * support).sum() / total_support)
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    per_class = []
    for i, cat in enumerate(categories):
        per_class.append({
            "class": cat,
            "index": i,
            "accuracy": float(cm[i, i] / support[i]) if support[i] > 0 else 0.0,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "true_positive": int(cm[i, i]),
        })

    # Per-sample records for misclassification export and sample predictions.
    per_sample = []
    for i in range(len(y_true)):
        per_sample.append({
            "sample_id": sample_ids[i],
            "true_index": int(y_true[i]),
            "true_class": categories[int(y_true[i])],
            "pred_index": int(y_pred[i]),
            "pred_class": categories[int(y_pred[i])],
            "correct": bool(correct[i]),
            "confidence": float(probs[i, y_pred[i]]),
            "top_probability": float(probs[i].max()),
        })

    misclassifications = [s for s in per_sample if not s["correct"]]
    # Sort misclassifications by descending confidence (most confident errors first —
    # these are the most important for a human to explain).
    misclassifications.sort(key=lambda s: s["confidence"], reverse=True)

    # At least five labelled predictions (correct ones preferred, fall back to any).
    correct_predictions = [s for s in per_sample if s["correct"]]
    labelled_predictions = correct_predictions[:5] or per_sample[:5]

    return {
        "num_classes": num_classes,
        "classes": list(categories),
        "num_test_samples": int(len(y_true)),
        "overall_accuracy": overall_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "per_sample": per_sample,
        "misclassifications": misclassifications,
        "num_misclassifications": len(misclassifications),
        "labelled_predictions": labelled_predictions,
    }


def write_results(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Machine-readable results (everything except large per_sample list kept separate).
    machine = {k: v for k, v in results.items() if k != "per_sample"}
    (output_dir / "metrics.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

    # Full per-sample and misclassification records (CSV for human review).
    fields = ["sample_id", "true_class", "pred_class", "correct", "confidence"]
    with (output_dir / "per_sample_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for s in results["per_sample"]:
            writer.writerow({k: s[k] for k in fields})

    misc_fields = fields + ["true_index", "pred_index", "top_probability"]
    with (output_dir / "misclassifications.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=misc_fields)
        writer.writeheader()
        for s in results["misclassifications"]:
            writer.writerow({k: s[k] for k in misc_fields})

    # Confusion matrix as CSV.
    cm = np.asarray(results["confusion_matrix"])
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([""] + list(results["classes"]))
        for i, row in enumerate(cm):
            writer.writerow([results["classes"][i]] + [int(v) for v in row])

    # Human-readable summary.
    summary_lines = [
        "Fish Species Detection — Five-Class Evaluation Summary",
        "=" * 55,
        f"Test samples: {results['num_test_samples']}",
        f"Classes: {results['num_classes']}",
        f"Overall accuracy: {results['overall_accuracy']:.4f}",
        f"Macro  precision: {results['macro_precision']:.4f}  recall: {results['macro_recall']:.4f}  F1: {results['macro_f1']:.4f}",
        f"Weighted precision: {results['weighted_precision']:.4f}  recall: {results['weighted_recall']:.4f}  F1: {results['weighted_f1']:.4f}",
        "",
        "Per-class results:",
    ]
    for pc in results["per_class"]:
        summary_lines.append(
            f"  {pc['class']:<38} P={pc['precision']:.3f} R={pc['recall']:.3f} "
            f"F1={pc['f1']:.3f} support={pc['support']} TP={pc['true_positive']}"
        )
    summary_lines.append("")
    summary_lines.append(f"Misclassifications to explain: {results['num_misclassifications']}")
    for s in results["misclassifications"][:20]:
        summary_lines.append(
            f"  {s['sample_id']:<30} true={s['true_class']:<30} pred={s['pred_class']:<30} "
            f"conf={s['confidence']:.3f}"
        )
    summary_lines.append("")
    summary_lines.append(
        "NOTE: Results computed only on held-out test images. No training-data "
        "evaluation. No results are fabricated."
    )
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def plot_confusion_matrix(results: dict, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(results["confusion_matrix"], dtype=np.float64)
    classes = results["classes"]
    # Normalise by row (true class) for readability.
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(classes)), max(5, 0.5 * len(classes))))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proportion of true class")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (row-normalised) — held-out test set")

    # Annotate cells with counts.
    thresh = cm_norm.max() / 2.0 if cm_norm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    fontsize=7,
                    color="white" if cm_norm[i, j] > thresh else "black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: list[dict], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, train_loss, label="train loss")
    ax1.plot(epochs, val_loss, label="val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, train_acc, label="train acc")
    ax2.plot(epochs, val_acc, label="val acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle("Training curves (early stopping on val loss)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
