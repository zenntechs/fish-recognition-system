"""Tests for the opt-in automated-screening preparation policy."""

from __future__ import annotations

import json
from pathlib import Path

from assignment.cli import _eligibility_gate_problems, _write_screening_audit
from assignment.data.manifest import _assess_eligibility
from assignment.data.review import ReviewManifest
from assignment.tests.conftest import CATEGORIES, _make_entry


def _assess(img: Path, *, policy: str, automated: str = "passed", screening=None,
            human: str = "pending", review: ReviewManifest | None = None):
    return _assess_eligibility(
        target_label="L1",
        abs_path=img,
        automated_status=automated,
        human_status=human,
        category=CATEGORIES[0],
        verified_human_statuses=frozenset({"verified"}),
        treat_needs_review_as_ineligible=True,
        notes="",
        review_manifest=review,
        sample_id="s1",
        review_policy=policy,
        screening=screening,
    )


def test_strict_policy_unchanged(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"x")
    assert _assess(img, policy="strict") == (False, "human_status='pending' (not yet human-verified)")
    ok, reason = _assess(img, policy="strict", human="verified")
    assert ok and reason == "eligible"


def test_automated_screening_is_provisional_and_not_human_verified(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"x")
    ok, reason = _assess(img, policy="automated-screening", screening=(True, "automated_screening=passed"))
    assert ok and "provisional" in reason and "not human-verified" in reason
    ok, _ = _assess(img, policy="automated-screening", screening=(False, "quality_flags=low_detail"))
    assert not ok


def test_needs_review_excluded_unless_genuinely_approved(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"x")
    screening = (True, "automated_screening=passed")
    assert not _assess(img, policy="automated-screening", automated="needs_review", screening=screening)[0]
    review = ReviewManifest()
    review.decide("s1", "approved", "Human", notes="checked visually")
    ok, reason = _assess(img, policy="automated-screening", automated="needs_review",
                         screening=screening, review=review)
    assert ok and reason == "human_review=approved"


def test_gate_reports_shortfall_and_imbalance(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"x")
    few = [_make_entry(f"s{i}", "L1", CATEGORIES[0], absolute_path=img) for i in range(3)]
    problems = _eligibility_gate_problems(few)
    assert any("shortfall 297" in p for p in problems)
    many = [_make_entry(f"a{i}", "L1", CATEGORIES[0], absolute_path=img) for i in range(300)]
    many += [_make_entry(f"b{i}", "L2", CATEGORIES[1], absolute_path=img) for i in range(10)]
    assert any("reasonably balanced" in p for p in _eligibility_gate_problems(many))


def test_audit_trail_distinguishes_auto_and_manual(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"x")
    auto = _make_entry("auto", "L1", CATEGORIES[0], absolute_path=img, eligible=True,
                       reason="automated_screening=passed (provisional; not human-verified)")
    manual = _make_entry("manual", "L1", CATEGORIES[0], absolute_path=img, eligible=True,
                         reason="human_review=approved")
    audit = _write_screening_audit([auto, manual], tmp_path / "out", "automated-screening")
    summary = json.loads((tmp_path / "out" / "screening_audit.json").read_text())
    assert summary["automated_screening_provisional"] == 1
    assert summary["manual_review_approved"] == 1
    assert summary["human_status_modified"] is False
    assert audit.exists()
