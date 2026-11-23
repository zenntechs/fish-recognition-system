"""Tests for the PyTorch dataset and transforms using synthetic images."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from assignment.data.dataset import FishCategoryDataset, train_transform, eval_transform
from assignment.tests.conftest import CATEGORIES, _make_entry


def _write_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(120, 60, 200)).save(path)


def test_dataset_len_and_class_counts(tmp_path):
    entries = []
    for i in range(3):
        img = tmp_path / f"img{i}.jpg"
        _write_image(img)
        entries.append(_make_entry(
            sample_id=f"s{i}", label=f"L{i}", category=CATEGORIES[i % 2],
            absolute_path=img,
        ))
    ds = FishCategoryDataset(entries=entries, categories=CATEGORIES, transform=eval_transform())
    assert len(ds) == 3
    counts = ds.class_counts()
    assert counts[CATEGORIES[0]] == 2
    assert counts[CATEGORIES[1]] == 1


def test_dataset_returns_tensor_and_index(tmp_path):
    img = tmp_path / "one.jpg"
    _write_image(img)
    entries = [_make_entry(sample_id="s1", label="L1", category=CATEGORIES[0], absolute_path=img)]
    ds = FishCategoryDataset(entries=entries, categories=CATEGORIES, transform=eval_transform())
    image, target, sample_id = ds[0]
    assert image.shape == (3, 224, 224)
    assert target == 0
    assert sample_id == "s1"


def test_dataset_rejects_unknown_category(tmp_path):
    img = tmp_path / "one.jpg"
    _write_image(img)
    entries = [_make_entry(sample_id="s1", label="L1", category="not_a_category", absolute_path=img)]
    with pytest.raises(ValueError):
        FishCategoryDataset(entries=entries, categories=CATEGORIES, transform=eval_transform())


def test_train_and_eval_transforms_differ(tmp_path):
    img = tmp_path / "one.jpg"
    _write_image(img)
    t_train = train_transform()
    t_eval = eval_transform()
    img_pil = Image.open(img).convert("RGB")
    out_train = t_train(img_pil)
    out_eval = t_eval(img_pil)
    assert out_train.shape == (3, 224, 224)
    assert out_eval.shape == (3, 224, 224)
    # Deterministic eval: two calls give the same result.
    assert torch.allclose(out_eval, t_eval(img_pil))
