"""Tests for manifest construction and validation using synthetic entries."""

from __future__ import annotations

from pathlib import Path

from assignment.data.manifest import ManifestEntry, manifest_summary, validate_manifest
from assignment.tests.conftest import CATEGORIES, _make_entry

_HERE = Path(__file__).resolve().parent


def _real_file() -> Path:
    return _HERE / "conftest.py"


def test_validate_manifest_accepts_well_formed_entries():
    entries = [
        _make_entry("a1", "L1", CATEGORIES[0], source_record_id="o1", absolute_path=_real_file()),
        _make_entry("a2", "L2", CATEGORIES[1], source_record_id="o2", absolute_path=_real_file()),
    ]
    assert validate_manifest(entries) == []


def test_validate_manifest_catches_duplicate_sample_id():
    entries = [
        _make_entry("dup", "L1", CATEGORIES[0], source_record_id="o1"),
        _make_entry("dup", "L2", CATEGORIES[1], source_record_id="o2"),
    ]
    problems = validate_manifest(entries)
    assert any("duplicate sample_id" in p for p in problems)


def test_validate_manifest_catches_eligible_without_category():
    entries = [
        _make_entry("a1", "L1", None, eligible=True, reason="should have category"),
    ]
    problems = validate_manifest(entries)
    assert any("eligible entry without category" in p for p in problems)


def test_manifest_summary_counts():
    entries = [
        _make_entry("a1", "L1", CATEGORIES[0], eligible=True),
        _make_entry("a2", "L1", CATEGORIES[0], eligible=True),
        _make_entry("a3", "L2", CATEGORIES[1], eligible=False, reason="human_status=pending"),
    ]
    summary = manifest_summary(entries)
    assert summary["total_images"] == 3
    assert summary["eligible_images"] == 2
    assert summary["per_category_eligible"][CATEGORIES[0]] == 2
    assert summary["per_category_eligible"].get(CATEGORIES[1], 0) == 0
