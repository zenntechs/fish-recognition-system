"""Tests for the standalone fish dataset collector (tools/collect_fish_samples.py)."""

from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path

import pytest
from PIL import Image

from tools import collect_fish_samples as cf


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_jpeg(seed: int, size: tuple[int, int] = (900, 700)) -> bytes:
    rng = random.Random(seed)
    image = Image.new("RGB", size)
    image.putdata(
        [tuple(rng.randrange(256) for _ in range(3)) for _ in range(size[0] * size[1])]
    )
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=92)
    return buffer.getvalue()


def observation(
    obs_id: int,
    photo_id: int,
    *,
    taxon: str = "Poecilia reticulata",
    ancestors: tuple[str, ...] = ("Poeciliidae", "Poecilia"),
    grade: str = "research",
    license_code: str = "cc-by",
    width: int = 1600,
    height: int = 1200,
    user: str = "Photographer",
) -> dict:
    return {
        "id": obs_id,
        "quality_grade": grade,
        "taxon": {"name": taxon},
        "ancestors": [{"name": name} for name in ancestors],
        "user": {"login": user.lower(), "name": user},
        "photos": [
            {
                "id": photo_id,
                "license_code": license_code,
                "url": (
                    "https://inaturalist-open-data.s3.amazonaws.com/"
                    f"photos/{photo_id}/square.jpg"
                ),
                "original_dimensions": {"width": width, "height": height},
                "attribution": f"(c) {user}, CC BY",
            }
        ],
    }


def candidate(photo_id: int, *, taxon: str = "Poecilia reticulata") -> cf.Candidate:
    return cf.Candidate(
        provider="iNaturalist",
        record_id=str(photo_id),
        page_url=f"https://www.inaturalist.org/observations/{photo_id}",
        download_url=f"https://example.invalid/{photo_id}.jpg",
        original_url=f"https://example.invalid/{photo_id}.jpg",
        license=cf.parse_license("CC BY 4.0"),
        author="Photographer",
        observed_scientific=taxon,
        width=1600,
        height=1200,
        taxon_confirmed=True,
        quality_grade="research",
    )


class FakeHttp:
    """Routable stand-in for HttpClient used by providers."""

    def __init__(
        self,
        *,
        taxa: list[dict] | None = None,
        observations: list[dict] | None = None,
        images: dict[str, bytes] | None = None,
    ) -> None:
        self.taxa = taxa if taxa is not None else [{"id": 1, "name": "Poecilia reticulata", "rank": "species"}]
        self.observations = observations or []
        self.images = images or {}
        self.calls: list[tuple[str, dict]] = []

    def get_json(self, url: str, params: dict | None = None) -> dict:
        self.calls.append((url, dict(params or {})))
        if url.endswith("/taxa"):
            return {"results": self.taxa}
        if url.endswith("/observations"):
            grade = (params or {}).get("quality_grade")
            if grade:
                return {"results": [o for o in self.observations if o["quality_grade"] == grade]}
            return {"results": list(self.observations)}
        if "commons.wikimedia.org" in url:
            return {"query": {"pages": {}}}
        raise AssertionError(f"unexpected URL {url}")

    def get_bytes(self, url: str) -> bytes:
        return self.images[url]


def new_store(tmp_path: Path) -> cf.DatasetStore:
    store = cf.DatasetStore(tmp_path / "SAMPLES FISH")
    store.ensure_layout()
    return store


def seed_image(store: cf.DatasetStore, label: str, target: cf.Target, seed: int = 0) -> cf.Row:
    return store.add_image(
        target=target,
        candidate=candidate(10_000 + seed, taxon=target.scientific),
        jpeg=make_jpeg(seed),
        width=900,
        height=700,
        date_accessed="2026-01-01",
    )


# ---------------------------------------------------------------------------
# targets, licences, taxonomy
# ---------------------------------------------------------------------------


def test_load_targets_reads_all_eighty_labels():
    targets = cf.load_targets()
    assert len(targets) == 80
    by_name = {t.label: t for t in targets}
    assert by_name["Clownfish"].scientific == "Amphiprion ocellaris"
    assert by_name["Butterflyfish"].rank == "genus"
    assert by_name["Cichlid"].rank == "family"
    assert by_name["Walleye"].rank == "species"


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("CC0", "CC0"),
        ("CC0 1.0", "CC0"),
        ("Public domain", "PD"),
        ("PD-USGov", "PD"),
        ("CC BY 4.0", "BY"),
        ("CC BY 2.0", "BY"),
        ("CC BY-SA 4.0", "BY-SA"),
        ("CC BY-SA 3.0", "BY-SA"),
    ],
)
def test_parse_license_accepts_permissive(raw, code):
    license = cf.parse_license(raw)
    assert license is not None and license.code == code


@pytest.mark.parametrize(
    "raw",
    ["CC BY-NC 4.0", "CC BY-ND 4.0", "CC BY-NC-SA 4.0", "All rights reserved", "", None, "unknown"],
)
def test_parse_license_rejects_noncommercial_and_unknown(raw):
    assert cf.parse_license(raw) is None


def test_share_alike_is_flagged():
    license = cf.parse_license("CC BY-SA 4.0")
    assert license is not None and license.share_alike is True


def test_taxon_matching_species_genus_family():
    species = cf.Target("Guppy", "Poecilia reticulata", "species")
    assert cf.taxon_matches(species, "Poecilia reticulata")
    assert not cf.taxon_matches(species, "Poecilia sphenops")

    genus = cf.Target("Butterflyfish", "Chaetodon", "genus")
    assert cf.taxon_matches(genus, "Chaetodon lunula")
    assert cf.taxon_matches(genus, "Chaetodon")
    assert not cf.taxon_matches(genus, "Scarus psittacus")

    family = cf.Target("Wrasse", "Labridae", "family")
    assert cf.taxon_matches(family, "Thalassoma lunare", ancestors=["Labridae"])
    assert not cf.taxon_matches(family, "Scarus psittacus", ancestors=["Scaridae"])


def test_candidate_problem_rejects_small_and_extreme_aspect():
    small = candidate(1)
    small.width, small.height = 200, 200
    assert "low resolution" in (cf.candidate_problem(small) or "")

    wide = candidate(2)
    wide.width, wide.height = 3000, 400
    assert "aspect ratio" in (cf.candidate_problem(wide) or "")

    fine = candidate(3)
    assert cf.candidate_problem(fine) is None


# ---------------------------------------------------------------------------
# store behaviour
# ---------------------------------------------------------------------------


def test_existing_image_detection(tmp_path):
    store = new_store(tmp_path)
    target = cf.Target("Guppy", "Poecilia reticulata", "species")
    seed_image(store, target.label, target, seed=1)
    store.save(target, store.rows)

    fresh = new_store(tmp_path)
    fresh.load()
    assert fresh.count("Guppy") == 1
    assert len(fresh.sha_index) == 1


def test_duplicate_prevention_by_hash_and_perceptual_hash(tmp_path):
    store = new_store(tmp_path)
    target = cf.Target("Guppy", "Poecilia reticulata", "species")
    payload = make_jpeg(7)
    store.add_image(
        target=target, candidate=candidate(1), jpeg=payload, width=900, height=700, date_accessed="2026-01-01"
    )
    with pytest.raises(cf.DuplicateImage):
        store.add_image(
            target=target, candidate=candidate(2), jpeg=payload, width=900, height=700, date_accessed="2026-01-01"
        )

    # Same picture resized and re-encoded: different bytes, near-identical hash.
    with Image.open(io.BytesIO(payload)) as image:
        resized = image.resize((700, 540))
        buffer = io.BytesIO()
        resized.save(buffer, "JPEG", quality=70)
    with pytest.raises(cf.DuplicateImage):
        store.add_image(
            target=target,
            candidate=candidate(3),
            jpeg=buffer.getvalue(),
            width=700,
            height=540,
            date_accessed="2026-01-01",
        )


def test_atomic_metadata_write_leaves_no_temp_files(tmp_path):
    store = new_store(tmp_path)
    target = cf.Target("Guppy", "Poecilia reticulata", "species")
    seed_image(store, target.label, target, seed=2)
    store.save(target, store.rows)

    assert not list(store.root.glob("*.tmp"))
    with store.metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert list(rows[0].keys()) == cf.METADATA_FIELDS


def test_exclusive_lock_blocks_second_writer(tmp_path):
    store = new_store(tmp_path)
    with store.lock():
        other = new_store(tmp_path)
        with pytest.raises(cf.LockHeld):
            with other.lock():
                pass


# ---------------------------------------------------------------------------
# HTTP behaviour
# ---------------------------------------------------------------------------


def test_http_429_honours_retry_after():
    responses = [(429, {"Retry-After": "0.5"}, b""), (200, {}, b'{"ok": true}')]
    sleeps: list[float] = []

    def fetch(url, user_agent, timeout):
        return responses.pop(0)

    client = cf.HttpClient(
        cf.HttpConfig(min_delay=0, max_delay=0, max_retries=3),
        fetch=fetch,
        sleep=sleeps.append,
        rng=random.Random(0),
    )
    assert client.get_json("https://example.invalid/api") == {"ok": True}
    assert sleeps and abs(sleeps[0] - 0.5) < 1e-6


def test_http_gives_up_after_bounded_retries():
    attempts: list[int] = []

    def fetch(url, user_agent, timeout):
        attempts.append(1)
        return 429, {}, b""

    client = cf.HttpClient(
        cf.HttpConfig(min_delay=0, max_delay=0, max_retries=2, max_backoff=0.01),
        fetch=fetch,
        sleep=lambda _: None,
        rng=random.Random(0),
    )
    with pytest.raises(cf.CollectorError):
        client.request("https://example.invalid/api")
    assert len(attempts) == 3  # initial attempt + two retries


def test_http_retries_server_errors():
    responses = [(503, {}, b""), (200, {}, b"payload")]

    def fetch(url, user_agent, timeout):
        return responses.pop(0)

    client = cf.HttpClient(
        cf.HttpConfig(min_delay=0, max_delay=0, max_retries=2, max_backoff=0.01),
        fetch=fetch,
        sleep=lambda _: None,
        rng=random.Random(0),
    )
    assert client.request("https://example.invalid/x") == b"payload"


# ---------------------------------------------------------------------------
# collection orchestration
# ---------------------------------------------------------------------------


def build_fake_http(photo_ids: list[int], *, taxon: str = "Poecilia reticulata") -> FakeHttp:
    observations = [
        observation(100 + i, photo_id, taxon=taxon, user=f"Photographer {i}")
        for i, photo_id in enumerate(photo_ids)
    ]
    images = {
        f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/large.jpg": make_jpeg(
            photo_id
        )
        for photo_id in photo_ids
    }
    return FakeHttp(observations=observations, images=images)


def test_collect_downloads_until_target_and_is_idempotent(tmp_path):
    store = new_store(tmp_path)
    targets = [cf.Target("Guppy", "Poecilia reticulata", "species")]
    http = build_fake_http([501, 502, 503, 504])

    first = cf.collect(store, targets, http=http, per_label=2, workers=1)
    assert first.added == 2
    assert store.count("Guppy") == 2

    second = cf.collect(store, targets, http=http, per_label=2, workers=1)
    assert second.added == 0
    assert store.count("Guppy") == 2


def test_collect_resumes_from_partial_dataset(tmp_path):
    store = new_store(tmp_path)
    target = cf.Target("Guppy", "Poecilia reticulata", "species")
    seed_image(store, target.label, target, seed=11)
    store.save(target, store.rows)

    http = build_fake_http([601, 602, 603])
    result = cf.collect(store, [target], http=http, per_label=3, workers=1)
    assert result.added == 2
    assert store.count("Guppy") == 3


def test_collect_respects_max_new_across_labels(tmp_path):
    store = new_store(tmp_path)
    targets = [
        cf.Target("Guppy", "Poecilia reticulata", "species"),
        cf.Target("Molly", "Poecilia sphenops", "species"),
    ]
    http = build_fake_http([701, 702, 703, 704])
    result = cf.collect(store, targets, http=http, per_label=2, max_new=2, workers=1)
    assert result.added == 2
    assert store.count("Guppy") == 2
    assert store.count("Molly") == 0


def test_collect_records_observed_species_for_genus(tmp_path):
    store = new_store(tmp_path)
    target = cf.Target("Butterflyfish", "Chaetodon", "genus")
    obs = observation(
        900,
        901,
        taxon="Chaetodon lunula",
        ancestors=("Chaetodontidae", "Chaetodon"),
    )
    http = FakeHttp(
        taxa=[{"id": 2, "name": "Chaetodon", "rank": "genus"}],
        observations=[obs],
        images={
            "https://inaturalist-open-data.s3.amazonaws.com/photos/901/large.jpg": make_jpeg(901)
        },
    )
    result = cf.collect(store, [target], http=http, per_label=1, workers=1)
    assert result.added == 1
    row = store.rows[0]
    assert row.data["target_scientific"] == "Chaetodon"
    assert row.data["observed_scientific"] == "Chaetodon lunula"
    assert row.data["human_status"] == "pending"


def test_collect_rejects_corrupted_download(tmp_path):
    store = new_store(tmp_path)
    target = cf.Target("Guppy", "Poecilia reticulata", "species")
    http = FakeHttp(
        observations=[observation(1, 2)],
        images={"https://inaturalist-open-data.s3.amazonaws.com/photos/2/large.jpg": b"not an image"},
    )
    result = cf.collect(store, [target], http=http, per_label=1, workers=1)
    assert result.added == 0
    assert result.rejected == 1
    assert store.rows == []


def test_collect_skips_duplicate_photos(tmp_path):
    store = new_store(tmp_path)
    target = cf.Target("Guppy", "Poecilia reticulata", "species")
    payload = make_jpeg(4242)
    url = "https://inaturalist-open-data.s3.amazonaws.com/photos/2/large.jpg"
    http = FakeHttp(
        observations=[observation(1, 2), observation(3, 4)],
        images={url: payload, "https://inaturalist-open-data.s3.amazonaws.com/photos/4/large.jpg": payload},
    )
    result = cf.collect(store, [target], http=http, per_label=2, workers=1)
    assert result.added == 1
    assert result.duplicates == 1


def test_dry_run_writes_nothing(tmp_path):
    store = new_store(tmp_path)
    target = cf.Target("Guppy", "Poecilia reticulata", "species")
    http = build_fake_http([801, 802])
    result = cf.collect(store, [target], http=http, per_label=2, workers=1, dry_run=True)
    assert result.added == 0
    assert not store.metadata_path.exists()
    assert store.count("Guppy") == 0


def test_low_resolution_candidates_are_filtered(tmp_path):
    store = new_store(tmp_path)
    target = cf.Target("Guppy", "Poecilia reticulata", "species")
    http = FakeHttp(
        observations=[observation(1, 2, width=300, height=200)],
        images={"https://inaturalist-open-data.s3.amazonaws.com/photos/2/large.jpg": make_jpeg(2)},
    )
    result = cf.collect(store, [target], http=http, per_label=1, workers=1)
    assert result.added == 0
    assert result.attempts == 0  # filtered before any download


# ---------------------------------------------------------------------------
# validation, status, contact sheets, repair
# ---------------------------------------------------------------------------


def test_validate_accepts_fresh_dataset(tmp_path):
    store = new_store(tmp_path)
    target = cf.Target("Guppy", "Poecilia reticulata", "species")
    seed_image(store, target.label, target, seed=21)
    store.save(target, store.rows)
    assert cf.validate_dataset(store.root) == []


def test_validate_detects_checksum_and_missing_file(tmp_path):
    store = new_store(tmp_path)
    target = cf.Target("Guppy", "Poecilia reticulata", "species")
    row = seed_image(store, target.label, target, seed=22)
    store.save(target, store.rows)

    path = store.root / row.path
    path.write_bytes(make_jpeg(999))
    issues = cf.validate_dataset(store.root)
    assert any("sha256 mismatch" in issue for issue in issues)

    path.unlink()
    issues = cf.validate_dataset(store.root)
    assert any("missing file" in issue for issue in issues)


def test_status_report_counts_labels(tmp_path):
    store = new_store(tmp_path)
    target = cf.Target("Guppy", "Poecilia reticulata", "species")
    seed_image(store, target.label, target, seed=23)
    store.save(target, store.rows)
    report = cf.status_report(store, [target])
    assert "| Guppy | 1 | pending |" in report
    assert "Total images: 1" in report


def test_contact_sheet_generation(tmp_path):
    store = new_store(tmp_path)
    target = cf.Target("Guppy", "Poecilia reticulata", "species")
    seed_image(store, target.label, target, seed=24)
    store.save(target, store.rows)
    out = tmp_path / "sheets"
    produced = cf.make_contact_sheets(store.root, [target], out)
    assert len(produced) == 1 and produced[0].exists()
    with Image.open(produced[0]) as sheet:
        assert sheet.width >= 400


def test_repair_drops_rows_without_files(tmp_path):
    store = new_store(tmp_path)
    target = cf.Target("Guppy", "Poecilia reticulata", "species")
    row = seed_image(store, target.label, target, seed=25)
    store.save(target, store.rows)
    (store.root / row.path).unlink()

    dry = cf.repair_metadata(store, apply=False)
    assert dry == [row.sample_id]
    assert len(store.rows) == 1

    applied = cf.repair_metadata(store, apply=True)
    assert applied == [row.sample_id]
    assert len(store.rows) == 0


def test_cli_validate_and_status(tmp_path, capsys):
    store = new_store(tmp_path)
    target = cf.Target("Guppy", "Poecilia reticulata", "species")
    seed_image(store, target.label, target, seed=26)
    store.save(target, store.rows)

    exit_code = cf.main(["--dataset", str(store.root), "validate"])
    assert exit_code == 0
    assert "images validated" in capsys.readouterr().out

    exit_code = cf.main(["--dataset", str(store.root), "status"])
    assert exit_code == 0
    assert "| Guppy | 1 |" in capsys.readouterr().out


def test_metadata_rows_preserve_existing_entries(tmp_path):
    store = new_store(tmp_path)
    target = cf.Target("Guppy", "Poecilia reticulata", "species")
    seed_image(store, target.label, target, seed=27)
    store.save(target, store.rows)

    http = build_fake_http([901, 902, 903])
    cf.collect(store, [target], http=http, per_label=2, workers=1)

    with store.metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert len({row["sample_id"] for row in rows}) == 2
    assert len({row["relative_path"] for row in rows}) == 2


def test_attribution_section_is_not_duplicated(tmp_path):
    store = new_store(tmp_path)
    target = cf.Target("Guppy", "Poecilia reticulata", "species")
    row = seed_image(store, target.label, target, seed=28)
    store.save(target, [row])
    store.save(target, [row])
    text = store.attribution_path.read_text(encoding="utf-8")
    assert text.count("## Guppy (`Poecilia reticulata`)") == 1
    assert text.count(row.sample_id) == 1
