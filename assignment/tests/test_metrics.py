"""Tests for evaluation metrics and confusion matrix (pure NumPy logic)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from assignment.evaluate import _confusion_matrix, _precision_recall_f1, evaluate_model


class TinyDS(Dataset):
    def __init__(self, images, targets, ids):
        self.images = images
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.images[i], self.targets[i], self.ids[i]


def test_confusion_matrix_counts():
    y_true = torch.tensor([0, 0, 1, 1, 2, 2])
    y_pred = torch.tensor([0, 1, 0, 1, 2, 1])
    cm = _confusion_matrix(y_true.numpy(), y_pred.numpy(), 3)
    expected = [[1, 1, 0], [1, 1, 0], [0, 1, 1]]
    assert cm.tolist() == expected


def test_precision_recall_f1_perfect():
    cm = torch.tensor([[3, 0, 0], [0, 4, 0], [0, 0, 2]], dtype=torch.int64).numpy()
    p, r, f, support = _precision_recall_f1(cm)
    assert list(p) == [1.0, 1.0, 1.0]
    assert list(r) == [1.0, 1.0, 1.0]
    assert list(f) == [1.0, 1.0, 1.0]
    assert list(support) == [3, 4, 2]


def test_precision_recall_f1_imperfect():
    # Class 0: TP=2, FP=1 (pred as 0 from true 1), FN=1 (true 0 pred as 1)
    cm = torch.tensor([[2, 1, 0], [1, 3, 0], [0, 0, 1]], dtype=torch.int64).numpy()
    p, r, f, support = _precision_recall_f1(cm)
    # class 0: precision = 2/(2+1)=0.667, recall = 2/(2+1)=0.667, f1=0.667
    assert abs(p[0] - 2 / 3) < 1e-6
    assert abs(r[0] - 2 / 3) < 1e-6
    assert abs(f[0] - 2 / 3) < 1e-6
    # class 2: perfect
    assert p[2] == 1.0 and r[2] == 1.0 and f[2] == 1.0
    assert list(support) == [3, 4, 1]


def test_zero_division_class_with_no_predictions():
    cm = torch.tensor([[0, 0], [0, 5]], dtype=torch.int64).numpy()  # class 0 never predicted
    p, r, f, support = _precision_recall_f1(cm)
    assert p[0] == 0.0   # no predicted positives -> precision 0
    assert r[0] == 0.0   # no true positives -> recall 0
    assert f[0] == 0.0
    assert support[0] == 0


def test_evaluate_model_end_to_end_cpu():
    # Tiny dummy model: always predict class index 0, bypassing conv layers.
    class Dummy(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[1.0, 0.0, 0.0]] * x.size(0))

    model = Dummy()
    model.eval()

    # 6 samples: true classes [0,0,1,1,2,2], images are 3-d vectors.
    images = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=torch.float32)
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    loader = DataLoader(
        TinyDS(images, targets, ["a", "b", "c", "d", "e", "f"]),
        batch_size=3, shuffle=False,
    )

    results = evaluate_model(model, loader, ["c0", "c1", "c2"], device="cpu")
    assert results["num_test_samples"] == 6
    # Always predicting class 0: correct only for the two true-0 samples.
    assert abs(results["overall_accuracy"] - (2 / 6)) < 1e-6
    assert results["num_misclassifications"] == 4
    assert len(results["labelled_predictions"]) >= 2
    assert results["confusion_matrix"] == [[2, 0, 0], [2, 0, 0], [2, 0, 0]]
