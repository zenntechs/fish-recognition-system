"""Tests for deterministic, group-aware, stratified splitting."""

from __future__ import annotations

from assignment.data.split import split_entries, SPLIT_NAMES
from assignment.tests.conftest import CATEGORIES, balanced_entries, single_group_entries


def test_split_uses_all_entries():
    entries = balanced_entries(__import__("pathlib").Path("/tmp"))
    result = split_entries(entries, seed=7)
    total = sum(len(result.entries_for(s)) for s in SPLIT_NAMES)
    assert total == len(entries)


def test_no_group_leakage_across_splits():
    entries = balanced_entries(__import__("pathlib").Path("/tmp"))
    result = split_entries(entries, seed=7)
    assert result.group_leakage_check["no_group_in_multiple_splits"] is True
    assert result.group_leakage_check["overlapping_group_keys_across_splits"] == []


def test_shared_observation_stays_in_one_split():
    """Both images of a duplicated observation must land in the same split."""
    entries = balanced_entries(__import__("pathlib").Path("/tmp"))
    result = split_entries(entries, seed=7)
    # For each observation id with 2 images, check both images share a split.
    from collections import defaultdict

    obs_to_splits = defaultdict(set)
    for a in result.assignments:
        if a.group_key and a.group_key.startswith("obs_"):
            obs_to_splits[a.group_key].add(a.split)
    for obs, splits in obs_to_splits.items():
        assert len(splits) == 1, f"observation {obs} leaked across {splits}"


def test_split_is_deterministic():
    entries = balanced_entries(__import__("pathlib").Path("/tmp"))
    r1 = split_entries(entries, seed=42)
    r2 = split_entries(entries, seed=42)
    for s in SPLIT_NAMES:
        ids_1 = [e.sample_id for e in r1.entries_for(s)]
        ids_2 = [e.sample_id for e in r2.entries_for(s)]
        assert ids_1 == ids_2, f"{s} split not deterministic"


def test_different_seed_gives_different_split():
    entries = balanced_entries(__import__("pathlib").Path("/tmp"))
    r1 = split_entries(entries, seed=1)
    r2 = split_entries(entries, seed=2)
    different = any(
        [e.sample_id for e in r1.entries_for(s)] != [e.sample_id for e in r2.entries_for(s)]
        for s in SPLIT_NAMES
    )
    assert different


def test_ratios_approximate_70_15_15():
    entries = balanced_entries(__import__("pathlib").Path("/tmp"))
    result = split_entries(entries, seed=7)
    counts = result.per_split_counts
    total = sum(counts.values())
    for name, ratio in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        share = counts[name] / total
        assert abs(share - ratio) <= 0.25, f"{name} share {share:.2f} far from {ratio}"


def test_every_category_present_in_train_when_sufficient():
    entries = balanced_entries(__import__("pathlib").Path("/tmp"))
    result = split_entries(entries, seed=7)
    for category in CATEGORIES:
        train_cats = [e.category for e in result.train]
        assert category in train_cats, f"{category} missing from train"
