"""Pytest fixtures: synthetic manifest entries for isolated testing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assignment.data.manifest import ManifestEntry

CATEGORIES = [
    "cartilaginous_fishes",
    "freshwater_bony_fishes",
    "marine_reef_bony_fishes",
    "marine_openwater_bottom_bony_fishes",
    "diadromous_euryhaline_bony_fishes",
]


def _make_entry(
    sample_id: str,
    label: str,
    category: str,
    source_record_id: str | None = None,
    eligible: bool = True,
    reason: str = "eligible",
    absolute_path: Path | None = None,
    observation_id: str | None = None,
    human_status: str = "verified",
    automated_status: str = "passed",
) -> ManifestEntry:
    # absolute_path defaults to a nonexistent path; callers that validate file
    # existence should pass a real tmp_path file.
    return ManifestEntry(
        sample_id=sample_id,
        target_label=label,
        target_scientific="Genus species",
        observed_scientific="Genus species",
        relative_path=f"{label}/{sample_id}.jpg",
        absolute_path=absolute_path or (Path("/tmp/nonexistent") / f"{sample_id}.jpg"),
        source="synthetic",
        source_record_id=source_record_id,
        observation_id=observation_id,
        source_page_url=None,
        license=None,
        author=None,
        sha256=None,
        width=224,
        height=224,
        bytes=1000,
        automated_status=automated_status,
        human_status=human_status,
        reviewer=None,
        reviewed_at=None,
        notes="",
        category=category,
        eligible=eligible,
        eligibility_reason=reason,
    )


def balanced_entries(tmp_path: Path) -> list[ManifestEntry]:
    """5 categories x 2 labels x 4 images; two images per label share an
    observation id (group leakage scenario)."""
    entries: list[ManifestEntry] = []
    for cat_idx, category in enumerate(CATEGORIES):
        for lbl in range(2):
            label = f"{category}_label{lbl}"
            obs_id = f"obs_{cat_idx}_{lbl}"
            for img in range(4):
                # first two images share an observation (duplicate source), last two are unique.
                src = obs_id if img < 2 else f"unique_{cat_idx}_{lbl}_{img}"
                entries.append(_make_entry(
                    sample_id=f"{label}_img{img}",
                    label=label,
                    category=category,
                    source_record_id=src,
                ))
    return entries


def single_group_entries(tmp_path: Path) -> list[ManifestEntry]:
    """One category, one observation, four near-duplicate images."""
    return [
        _make_entry(f"solo_img{i}", "solo_label", CATEGORIES[1], source_record_id="obs_solo")
        for i in range(4)
    ]
