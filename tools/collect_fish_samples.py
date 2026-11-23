#!/usr/bin/env python3
"""Collect a redistributable fish image dataset from open APIs.

The collector fills ``SAMPLES FISH/`` so that every label declared in
``app/species.py`` has at least four valid photographs, using iNaturalist as
the primary source and Wikimedia Commons as the fallback.

It is deliberately standalone: no web application code is imported, no model
is loaded and no AI service is contacted.  Run it from a terminal:

    python tools/collect_fish_samples.py collect --max-new 2   # smoke test
    python tools/collect_fish_samples.py collect               # full run
    python tools/collect_fish_samples.py status
    python tools/collect_fish_samples.py validate
    python tools/collect_fish_samples.py contact-sheets

Design notes
------------
* One writer.  Downloads may run concurrently, but every filesystem mutation
  happens on the main thread while holding an advisory lock on the dataset.
* Atomic writes.  Metadata, checksums and attribution are written to a sibling
  temporary file and moved into place with :func:`os.replace`.
* Resumable.  Per-label page cursors and failures live in
  ``SAMPLES FISH/.collector/state.json``.
* Honest provenance.  Automated taxonomy matching is not human review, so new
  rows are ``human_status=pending`` and uncertain matches are
  ``automated_status=needs_review``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime as dt
import fcntl
import hashlib
import importlib.util
import io
import json
import logging
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "SAMPLES FISH"
SPECIES_FILE = REPO_ROOT / "app" / "species.py"

MAX_EDGE = 1280
JPEG_QUALITY = 85
MIN_SOURCE_EDGE = 400
DHASH_DISTANCE = 6
DEFAULT_TARGET = 4
DEFAULT_WORKERS = 2

METADATA_FIELDS = [
    "sample_id",
    "target_label",
    "target_scientific",
    "observed_scientific",
    "relative_path",
    "source",
    "source_record_id",
    "source_page_url",
    "original_image_url",
    "license",
    "license_url",
    "attribution",
    "author",
    "date_accessed",
    "sha256",
    "width",
    "height",
    "bytes",
    "automated_status",
    "human_status",
    "reviewer",
    "reviewed_at",
    "notes",
]

USER_AGENT = (
    "AquaVision-Dataset-Collector/1.0 "
    "(https://github.com/zenntechs/fish-recognition-system; open dataset)"
)

log = logging.getLogger("fishcollect")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CollectorError(RuntimeError):
    """Recoverable collection failure."""


class DuplicateImage(CollectorError):
    """Candidate duplicates an image already in the dataset."""


class RateLimited(CollectorError):
    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__("rate limited")
        self.retry_after = retry_after


class HttpStatusError(CollectorError):
    def __init__(self, status: int, url: str) -> None:
        super().__init__(f"HTTP {status} for {url}")
        self.status = status


class ServerError(CollectorError):
    pass


class LockHeld(CollectorError):
    pass


# ---------------------------------------------------------------------------
# Targets and taxonomy
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Target:
    label: str
    scientific: str
    rank: str  # "species" | "genus" | "family"

    @property
    def slug(self) -> str:
        return slugify(self.scientific)

    @property
    def display(self) -> str:
        return f"{self.label} ({self.scientific})"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def infer_rank(scientific: str) -> str:
    name = scientific.strip()
    if name.endswith("idae"):
        return "family"
    if len(name.split()) >= 2:
        return "species"
    return "genus"


def _load_species_module(path: Path):
    spec = importlib.util.spec_from_file_location("_fish_species", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise CollectorError(f"cannot load species list from {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution so dataclasses can resolve the module by name.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def load_targets(species_file: Path = SPECIES_FILE) -> list[Target]:
    """Read the authoritative 80 labels from ``app/species.py``."""
    module = _load_species_module(species_file)
    targets = [
        Target(label=s.name, scientific=s.scientific, rank=infer_rank(s.scientific))
        for s in module.SPECIES
    ]
    if not targets:
        raise CollectorError("no species labels found")
    return targets


def normalize_taxon(name: str | None) -> str:
    if not name:
        return ""
    text = re.sub(r"\(.*?\)", " ", name)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def taxon_matches(
    target: Target,
    observed: str | None,
    ancestors: Sequence[str] = (),
) -> bool:
    """Return True when ``observed`` is confirmed to belong to ``target``.

    ``ancestors`` are higher-taxon names (for iNaturalist, the observation's
    ancestor chain, including its family/genus).
    """
    leaf = normalize_taxon(observed)
    ancestor_set = {normalize_taxon(a) for a in ancestors if a}
    want = normalize_taxon(target.scientific)

    if not leaf:
        return False
    if target.rank == "species":
        return leaf == want or leaf.startswith(want + " ")
    if target.rank == "genus":
        return leaf.split(" ")[0] == want or want in ancestor_set
    # family
    return leaf == want or want in ancestor_set


# ---------------------------------------------------------------------------
# Licences
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class License:
    code: str  # CC0 | PD | BY | BY-SA
    name: str
    url: str
    share_alike: bool = False


_LICENSE_URL = {
    "CC0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "PD": "https://creativecommons.org/publicdomain/mark/1.0/",
    "BY": "https://creativecommons.org/licenses/by/4.0/",
    "BY-SA": "https://creativecommons.org/licenses/by-sa/4.0/",
}


def _version(text: str, default: str = "4.0") -> str:
    match = re.search(r"(\d+\.\d+)", text)
    return match.group(1) if match else default


def parse_license(raw: str | None) -> License | None:
    """Map a provider licence string to an accepted licence, else ``None``.

    Rejects NonCommercial, NoDerivatives and anything unrecognised.
    """
    if not raw:
        return None
    text = str(raw).strip()
    low = text.lower().replace("_", " ").replace("-", " ")
    tokens = re.findall(r"[a-z0-9.]+", low)
    token_set = set(tokens)

    if "nc" in token_set or "nd" in token_set:
        return None
    if "noncommercial" in low or "noderiv" in low:
        return None
    if "cc0" in token_set or "zero" in token_set:
        return License("CC0", "CC0 1.0", _LICENSE_URL["CC0"])
    if ("public" in token_set and "domain" in token_set) or "pd" in token_set:
        return License("PD", "Public domain", _LICENSE_URL["PD"])
    version = _version(text)
    if "sa" in token_set or "sharealike" in low:
        return License(
            "BY-SA",
            f"CC BY-SA {version}",
            f"https://creativecommons.org/licenses/by-sa/{version}/",
            share_alike=True,
        )
    if "by" in token_set or "attribution" in token_set:
        return License(
            "BY",
            f"CC BY {version}",
            f"https://creativecommons.org/licenses/by/{version}/",
        )
    return None


# ---------------------------------------------------------------------------
# HTTP with pacing, retries and 429 handling
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class HttpConfig:
    user_agent: str = USER_AGENT
    timeout: float = 60.0
    max_retries: int = 4
    min_delay: float = 1.0
    max_delay: float = 2.5
    backoff: float = 2.0
    max_backoff: float = 30.0


def _urllib_fetch(url: str, user_agent: str, timeout: float) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(
        url, headers={"User-Agent": user_agent, "Accept": "*/*"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        body = b""
        with contextlib.suppress(Exception):
            body = exc.read()
        headers = dict(exc.headers or {})
        return exc.code, headers, body


class HttpClient:
    """Small JSON/binary client shared by both providers."""

    def __init__(
        self,
        config: HttpConfig | None = None,
        fetch: Callable[[str, str, float], tuple[int, dict, bytes]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or HttpConfig()
        self._fetch = fetch or _urllib_fetch
        self._sleep = sleep
        self._clock = clock
        self._rng = rng or random.Random()
        self._lock = __import__("threading").Lock()
        self._last_request = 0.0

    def _pace(self) -> None:
        with self._lock:
            pause = self._rng.uniform(self.config.min_delay, self.config.max_delay)
            elapsed = self._clock() - self._last_request
            if elapsed < pause:
                self._sleep(pause - elapsed)
            self._last_request = self._clock()

    def _retry_delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after:
            return min(retry_after, self.config.max_backoff)
        base = min(self.config.backoff ** attempt, self.config.max_backoff)
        return base + self._rng.uniform(0, 1)

    def request(self, url: str, params: dict | None = None) -> bytes:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
        attempt = 0
        while True:
            attempt += 1
            self._pace()
            try:
                status, headers, body = self._fetch(
                    url, self.config.user_agent, self.config.timeout
                )
                if status == 429:
                    retry_after = None
                    raw = headers.get("Retry-After") or headers.get("retry-after")
                    with contextlib.suppress(TypeError, ValueError):
                        retry_after = float(raw)  # type: ignore[arg-type]
                    raise RateLimited(retry_after)
                if status >= 500:
                    raise ServerError(f"HTTP {status} for {url}")
                if status >= 400:
                    raise HttpStatusError(status, url)
                return body
            except RateLimited as exc:
                if attempt > self.config.max_retries:
                    raise CollectorError(f"rate limited repeatedly: {url}") from exc
                delay = self._retry_delay(attempt, exc.retry_after)
                log.warning("429 from %s; sleeping %.1fs", short(url), delay)
                self._sleep(delay)
            except (ServerError, urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt > self.config.max_retries:
                    raise CollectorError(f"request failed: {url}: {exc}") from exc
                delay = self._retry_delay(attempt, None)
                log.warning("retry %d for %s after %.1fs: %s", attempt, short(url), delay, exc)
                self._sleep(delay)

    def get_json(self, url: str, params: dict | None = None) -> dict:
        body = self.request(url, params)
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CollectorError(f"invalid JSON from {short(url)}") from exc

    def get_bytes(self, url: str) -> bytes:
        return self.request(url)


def short(url: str, limit: int = 90) -> str:
    return url if len(url) <= limit else url[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Candidate:
    provider: str
    record_id: str
    page_url: str
    download_url: str
    original_url: str
    license: License
    author: str
    observed_scientific: str
    width: int
    height: int
    taxon_confirmed: bool
    quality_grade: str = ""
    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.record_id}"


def candidate_problem(c: Candidate) -> str | None:
    """Return a rejection reason for clearly unsuitable candidates."""
    if min(c.width, c.height) < MIN_SOURCE_EDGE:
        return f"low resolution ({c.width}x{c.height})"
    if c.width <= 0 or c.height <= 0:
        return "missing dimensions"
    ratio = max(c.width, c.height) / min(c.width, c.height)
    if ratio > 4.0:
        return f"extreme aspect ratio ({ratio:.1f})"
    banned = ("watermark", "drawing", "illustration", "diagram", "logo")
    haystack = f"{c.page_url} {c.original_url}".lower()
    for word in banned:
        if word in haystack:
            return f"unsuitable keyword: {word}"
    return None


# ---------------------------------------------------------------------------
# Provider: iNaturalist
# ---------------------------------------------------------------------------


INAT_API = "https://api.inaturalist.org/v1"
INAT_PHOTO_HOST = "inaturalist-open-data.s3.amazonaws.com"


class INaturalistProvider:
    name = "iNaturalist"

    def __init__(
        self,
        http: HttpClient,
        *,
        per_page: int = 200,
        max_pages: int = 2,
    ) -> None:
        self.http = http
        self.per_page = per_page
        self.max_pages = max_pages
        self._taxon_cache: dict[str, dict | None] = {}

    def resolve_taxon(self, target: Target) -> dict | None:
        if target.scientific in self._taxon_cache:
            return self._taxon_cache[target.scientific]
        data = self.http.get_json(
            f"{INAT_API}/taxa", {"q": target.scientific, "per_page": 20}
        )
        results = data.get("results") or []
        want = normalize_taxon(target.scientific)
        exact = [r for r in results if normalize_taxon(r.get("name")) == want]
        pool = exact or results
        chosen = None
        for item in pool:
            if item.get("rank") == target.rank:
                chosen = item
                break
        if chosen is None and pool:
            chosen = pool[0]
        self._taxon_cache[target.scientific] = chosen
        return chosen

    def candidates(self, target: Target, limit: int = 24) -> list[Candidate]:
        taxon = self.resolve_taxon(target)
        if not taxon:
            log.warning("iNaturalist has no taxon for %s", target.scientific)
            return []

        found: list[Candidate] = []
        seen_photos: set[str] = set()
        for grade in ("research", None):
            page = 0
            while len(found) < limit and page < self.max_pages:
                page += 1
                params: dict[str, object] = {
                    "taxon_id": taxon["id"],
                    "photo_license": "cc0,cc-by,cc-by-sa",
                    "per_page": self.per_page,
                    "page": page,
                    "order_by": "votes",
                }
                if grade:
                    params["quality_grade"] = grade
                try:
                    data = self.http.get_json(f"{INAT_API}/observations", params)
                except CollectorError as exc:
                    log.warning("iNaturalist observation page failed: %s", exc)
                    break
                results = data.get("results") or []
                if not results:
                    break
                for obs in results:
                    quality = str(obs.get("quality_grade") or "")
                    if grade is None and quality == "research":
                        continue  # already covered by the first pass
                    obs_taxon = obs.get("taxon") or {}
                    ancestors = [
                        a.get("name", "") for a in (obs.get("ancestors") or [])
                    ]
                    if not taxon_matches(target, obs_taxon.get("name"), ancestors):
                        continue
                    author = (obs.get("user") or {}).get("name") or (
                        obs.get("user") or {}
                    ).get("login", "")
                    for photo in obs.get("photos") or []:
                        photo_id = str(photo.get("id") or "")
                        if not photo_id or photo_id in seen_photos:
                            continue
                        license = parse_license(photo.get("license_code"))
                        if license is None:
                            continue
                        url = str(photo.get("url") or "")
                        if INAT_PHOTO_HOST not in url:
                            continue
                        ext = url.rsplit(".", 1)[-1].split("?")[0].lower() or "jpg"
                        if ext not in {"jpg", "jpeg", "png"}:
                            continue
                        dims = photo.get("original_dimensions") or {}
                        seen_photos.add(photo_id)
                        found.append(
                            Candidate(
                                provider=self.name,
                                record_id=photo_id,
                                page_url=f"https://www.inaturalist.org/observations/{obs.get('id')}",
                                download_url=(
                                    f"https://{INAT_PHOTO_HOST}/photos/{photo_id}/large.{ext}"
                                ),
                                original_url=(
                                    f"https://{INAT_PHOTO_HOST}/photos/{photo_id}/large.{ext}"
                                ),
                                license=license,
                                author=author,
                                observed_scientific=str(obs_taxon.get("name") or ""),
                                width=int(dims.get("width") or 0),
                                height=int(dims.get("height") or 0),
                                taxon_confirmed=True,
                                quality_grade=quality,
                                notes=f"Observation {obs.get('id')}.",
                            )
                        )
                        if len(found) >= limit:
                            break
                    if len(found) >= limit:
                        break
        return found


# ---------------------------------------------------------------------------
# Provider: Wikimedia Commons
# ---------------------------------------------------------------------------


COMMONS_API = "https://commons.wikimedia.org/w/api.php"


def _strip_html(value: object) -> str:
    text = re.sub(r"<[^>]+>", "", str(value or ""))
    text = text.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    return re.sub(r"\s+", " ", text).strip()


def _commons_thumb(original: str, width: int = MAX_EDGE) -> str:
    """Build a thumbnail URL for a Commons original."""
    parts = urllib.parse.urlsplit(original)
    if "/wikipedia/commons/" not in parts.path:
        return original
    head, _, filename = parts.path.rpartition("/")
    if not filename:
        return original
    thumb_head = head.replace("/wikipedia/commons/", "/wikipedia/commons/thumb/", 1)
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, f"{thumb_head}/{filename}/{width}px-{filename}", "", "")
    )


class CommonsProvider:
    name = "Wikimedia Commons"

    def __init__(self, http: HttpClient, *, per_page: int = 50) -> None:
        self.http = http
        self.per_page = per_page

    def candidates(self, target: Target, limit: int = 24) -> list[Candidate]:
        categories = [f"Category:{target.scientific}"]
        found: list[Candidate] = []
        seen: set[str] = set()
        for category in categories:
            try:
                data = self.http.get_json(
                    COMMONS_API,
                    {
                        "action": "query",
                        "format": "json",
                        "generator": "categorymembers",
                        "gcmtitle": category,
                        "gcmtype": "file",
                        "gcmlimit": str(min(self.per_page, 100)),
                        "prop": "imageinfo",
                        "iiprop": "url|extmetadata|size|mime",
                        "iiurlwidth": str(MAX_EDGE),
                    },
                )
            except CollectorError as exc:
                log.warning("Commons category %s failed: %s", category, exc)
                continue
            pages = (data.get("query") or {}).get("pages") or {}
            for page in pages.values():
                info = (page.get("imageinfo") or [{}])[0]
                if info.get("mime") not in {"image/jpeg", "image/png"}:
                    continue
                meta = info.get("extmetadata") or {}
                license = parse_license(_strip_html(meta.get("LicenseShortName", {}).get("value")))
                if license is None:
                    continue
                original = str(info.get("url") or "")
                if not original or original in seen:
                    continue
                seen.add(original)
                author = _strip_html(meta.get("Artist", {}).get("value")) or "Unknown"
                title = str(page.get("title") or "")
                found.append(
                    Candidate(
                        provider=self.name,
                        record_id=str(page.get("pageid") or title),
                        page_url=str(info.get("descriptionurl") or ""),
                        download_url=str(info.get("thumburl") or _commons_thumb(original)),
                        original_url=original.split("?")[0],
                        license=license,
                        author=author,
                        observed_scientific=target.scientific,
                        width=int(info.get("width") or 0),
                        height=int(info.get("height") or 0),
                        taxon_confirmed=False,
                        quality_grade="",
                        notes=f"{title}; Commons category match, taxonomy not independently verified.",
                    )
                )
                if len(found) >= limit:
                    return found
        return found


# ---------------------------------------------------------------------------
# Image processing, hashes and duplicate checks
# ---------------------------------------------------------------------------


def process_image(raw: bytes) -> tuple[bytes, int, int]:
    """Decode ``raw``, orient, resize, re-encode and strip metadata."""
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            oriented = ImageOps.exif_transpose(image)
            rgb = oriented.convert("RGB")
            rgb.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
            out = io.BytesIO()
            rgb.save(out, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=False)
            return out.getvalue(), rgb.width, rgb.height
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise CollectorError(f"image could not be decoded: {exc}") from exc


def dhash(raw: bytes) -> int:
    """64-bit difference hash, robust to resizing and mild re-encoding."""
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            small = ImageOps.exif_transpose(image).convert("L").resize(
                (9, 8), Image.LANCZOS
            )
            pixels = small.tobytes()
    except (UnidentifiedImageError, OSError) as exc:
        raise CollectorError(f"cannot hash image: {exc}") from exc
    bits = 0
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            bits = (bits << 1) | (1 if left < right else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def near_duplicate(hashes: Iterable[tuple[str, int]], candidate: int) -> str | None:
    for sample_id, value in hashes:
        if hamming(value, candidate) <= DHASH_DISTANCE:
            return sample_id
    return None


# ---------------------------------------------------------------------------
# Dataset store (single writer, atomic)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Row:
    data: dict[str, str]

    @property
    def sample_id(self) -> str:
        return self.data["sample_id"]

    @property
    def label(self) -> str:
        return self.data["target_label"]

    @property
    def path(self) -> str:
        return self.data["relative_path"]


class DatasetStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.metadata_path = self.root / "metadata.csv"
        self.checksums_path = self.root / "CHECKSUMS.sha256"
        self.attribution_path = self.root / "ATTRIBUTION.md"
        self.progress_path = self.root / "PROGRESS.md"
        self.state_dir = self.root / ".collector"
        self.state_path = self.state_dir / "state.json"
        self.error_log = self.state_dir / "errors.log"
        self.lock_path = self.state_dir / "lock"
        self.rows: list[Row] = []
        self.by_label: dict[str, list[Row]] = {}
        self.sha_index: dict[str, str] = {}
        self.dhash_index: list[tuple[str, int]] = []
        self._lock_handle: io.TextIOWrapper | None = None

    # -- lifecycle ---------------------------------------------------------

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def lock(self, blocking: bool = False):
        self.ensure_layout()
        handle = self.lock_path.open("a+")
        try:
            flags = fcntl.LOCK_EX
            if not blocking:
                flags |= fcntl.LOCK_NB
            try:
                fcntl.flock(handle, flags)
            except BlockingIOError as exc:
                raise LockHeld(
                    f"another collector holds {self.lock_path}"
                ) from exc
            self._lock_handle = handle
            yield self
        finally:
            self._lock_handle = None
            with contextlib.suppress(Exception):
                fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def load(self, strict: bool = False) -> None:
        """Load existing metadata and index the images that are still valid."""
        self.rows = []
        self.by_label = {}
        self.sha_index = {}
        self.dhash_index = []
        if self.metadata_path.exists():
            with self.metadata_path.open(newline="", encoding="utf-8") as handle:
                for raw in csv.DictReader(handle):
                    if not raw.get("sample_id") or not raw.get("relative_path"):
                        continue
                    self.rows.append(Row({k: (v or "") for k, v in raw.items()}))
        for row in self.rows:
            path = self.root / row.path
            if not path.exists():
                if strict:
                    raise CollectorError(f"missing file for {row.sample_id}: {row.path}")
                log.warning("metadata row without file: %s", row.sample_id)
                continue
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                file_hash = dhash(path.read_bytes())
            except (OSError, CollectorError) as exc:
                log.warning("cannot index %s: %s", row.sample_id, exc)
                continue
            row.data["sha256"] = row.data.get("sha256") or digest
            self.by_label.setdefault(row.label, []).append(row)
            self.sha_index.setdefault(digest, row.sample_id)
            self.dhash_index.append((row.sample_id, file_hash))

    # -- queries -----------------------------------------------------------

    def count(self, label: str) -> int:
        return len(self.by_label.get(label, []))

    def next_ordinal(self, label: str) -> int:
        highest = 0
        for row in self.by_label.get(label, []):
            match = re.search(r"_(\d{3})_", row.path)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    def label_dir(self, label: str) -> Path:
        return self.root / label

    # -- mutations ---------------------------------------------------------

    def add_image(
        self,
        *,
        target: Target,
        candidate: Candidate,
        jpeg: bytes,
        width: int,
        height: int,
        date_accessed: str,
    ) -> Row:
        digest = hashlib.sha256(jpeg).hexdigest()
        if digest in self.sha_index:
            raise DuplicateImage(f"exact duplicate of {self.sha_index[digest]}")
        image_hash = dhash(jpeg)
        twin = near_duplicate(self.dhash_index, image_hash)
        if twin:
            raise DuplicateImage(f"near-duplicate of {twin}")

        ordinal = self.next_ordinal(target.label)
        suffix = "inat" if candidate.provider == "iNaturalist" else "commons"
        filename = (
            f"{slugify(target.label)}_{ordinal:03d}_{target.slug}_"
            f"{suffix}-{candidate.record_id}.jpg"
        )
        relative = f"{target.label}/{filename}"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(destination, jpeg)

        automated = (
            "passed"
            if candidate.taxon_confirmed and candidate.quality_grade in ("research", "")
            else "needs_review"
        )
        notes = [
            candidate.notes.strip(),
            "Automated checks passed." if automated == "passed" else "Needs human review.",
            "Human verification pending. Resized to max 1280 px, JPEG q85, EXIF stripped.",
        ]
        if candidate.license.share_alike:
            notes.insert(0, "Share-alike: derivative distribution must use the same licence.")
        row = Row(
            {
                "sample_id": f"{slugify(target.label)}-{ordinal:03d}",
                "target_label": target.label,
                "target_scientific": target.scientific,
                "observed_scientific": candidate.observed_scientific,
                "relative_path": relative,
                "source": candidate.provider,
                "source_record_id": candidate.record_id,
                "source_page_url": candidate.page_url,
                "original_image_url": candidate.original_url,
                "license": candidate.license.name,
                "license_url": candidate.license.url,
                "attribution": f"{candidate.author} / {candidate.provider} / {candidate.license.name}",
                "author": candidate.author,
                "date_accessed": date_accessed,
                "sha256": digest,
                "width": str(width),
                "height": str(height),
                "bytes": str(len(jpeg)),
                "automated_status": automated,
                "human_status": "pending",
                "reviewer": "",
                "reviewed_at": "",
                "notes": " ".join(n for n in notes if n),
            }
        )
        self.rows.append(row)
        self.by_label.setdefault(target.label, []).append(row)
        self.sha_index[digest] = row.sample_id
        self.dhash_index.append((row.sample_id, image_hash))
        return row

    def write_metadata(self) -> None:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=METADATA_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in self.rows:
            writer.writerow(row.data)
        write_atomic(self.metadata_path, buffer.getvalue().encode("utf-8"))

    def rebuild_checksums(self) -> int:
        lines = []
        for row in self.rows:
            path = self.root / row.path
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {row.path}")
        write_atomic(self.checksums_path, ("\n".join(lines) + "\n").encode("utf-8"))
        return len(lines)

    def upsert_attribution(self, target: Target, rows: Sequence[Row]) -> None:
        section_header = f"## {target.label} (`{target.scientific}`)"
        entries = []
        for row in rows:
            extra = ""
            observed = row.data.get("observed_scientific", "")
            if observed and normalize_taxon(observed) != normalize_taxon(target.scientific):
                extra = f" Observed species: *{observed}*."
            entries.append(
                f"- **{row.sample_id}** — {row.data['author']}, "
                f"[{row.data['source']}]({row.data['source_page_url']}), "
                f"{row.data['license']}.{extra}"
            )
        block = section_header + "\n\n" + "\n".join(entries) + "\n"

        existing = self.attribution_path.read_text(encoding="utf-8") if self.attribution_path.exists() else ""
        pattern = re.compile(
            rf"^## {re.escape(target.label)} \(`[^`]+`\)\n.*?(?=^## |\Z)",
            re.MULTILINE | re.DOTALL,
        )
        if pattern.search(existing):
            updated = pattern.sub(block + "\n", existing, count=1)
        else:
            updated = existing.rstrip("\n") + "\n\n" + block
        write_atomic(self.attribution_path, updated.encode("utf-8"))

    def save(self, target: Target, new_rows: Sequence[Row]) -> None:
        """Persist metadata, checksums, attribution and progress together."""
        self.write_metadata()
        self.rebuild_checksums()
        self.upsert_attribution(target, new_rows)
        update_progress(self, targets_cache={})

    # -- state -------------------------------------------------------------

    def load_state(self) -> dict:
        if not self.state_path.exists():
            return {"version": 1, "labels": {}}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("state file unreadable; starting fresh")
            return {"version": 1, "labels": {}}

    def save_state(self, state: dict) -> None:
        state["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        write_atomic(
            self.state_path,
            json.dumps(state, indent=2, sort_keys=True).encode("utf-8"),
        )


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


def update_progress(store: DatasetStore, targets_cache: dict) -> None:
    label_counts: dict[str, int] = {}
    for row in store.rows:
        label_counts[row.label] = label_counts.get(row.label, 0) + 1
    complete = sorted(label for label in label_counts if label_counts[label] >= DEFAULT_TARGET)
    total = len(store.rows)

    text = store.progress_path.read_text(encoding="utf-8") if store.progress_path.exists() else "# Phase 1 Progress\n"
    text = re.sub(
        r"- Species with 4 images collected: .*",
        f"- Species with 4 images collected: {len(complete)} of 80",
        text,
    )
    text = re.sub(
        r"- Species human-verified: \d+ of 80",
        "- Species human-verified: 0 of 80",
        text,
    )
    text = re.sub(r"- Remaining species: \d+", f"- Remaining species: {80 - len(complete)}", text)
    text = re.sub(r"- Current total: \d+ images", f"- Current total: {total} images", text)
    text = re.sub(
        r"- Human verification: .*",
        "- Human verification: **pending for all images** "
        "(automated checks and agent inspection are not human review)",
        text,
    )
    text = re.sub(
        r"- Next action: .*",
        "- Next action: run `python tools/collect_fish_samples.py collect` "
        "until every label has four images, then review contact sheets",
        text,
    )

    section = ["## Automated collection", "", "| Label | Images | Status |", "|---|---:|---|"]
    for label, count in sorted(label_counts.items()):
        status = "complete" if count >= DEFAULT_TARGET else f"needs {DEFAULT_TARGET - count}"
        section.append(f"| {label} | {count} | {status} |")
    section.append("")
    block = "\n".join(section)

    marker = "## Automated collection"
    if marker in text:
        text = re.sub(rf"^{re.escape(marker)}\n.*\Z", block + "\n", text, count=1, flags=re.MULTILINE | re.DOTALL)
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    write_atomic(store.progress_path, text.encode("utf-8"))


# ---------------------------------------------------------------------------
# Collection orchestration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CollectResult:
    added: int = 0
    attempts: int = 0
    skipped: int = 0
    duplicates: int = 0
    rejected: int = 0
    per_label: dict[str, int] = dataclasses.field(default_factory=dict)


def iter_candidates(
    providers: Sequence[object],
    target: Target,
    *,
    limit: int,
) -> list[Candidate]:
    collected: list[Candidate] = []
    seen: set[str] = set()
    for provider in providers:
        try:
            batch = provider.candidates(target, limit=limit)  # type: ignore[attr-defined]
        except CollectorError as exc:
            log.warning("%s failed for %s: %s", provider.name, target.label, exc)  # type: ignore[attr-defined]
            continue
        for candidate in batch:
            if candidate.key in seen:
                continue
            seen.add(candidate.key)
            reason = candidate_problem(candidate)
            if reason:
                log.info("reject %s: %s", candidate.key, reason)
                continue
            collected.append(candidate)
    return collected


def fetch_candidates(
    http: HttpClient,
    candidates: Sequence[Candidate],
    *,
    workers: int,
) -> list[tuple[Candidate, bytes | None]]:
    def fetch(candidate: Candidate) -> tuple[Candidate, bytes | None]:
        try:
            return candidate, http.get_bytes(candidate.download_url)
        except CollectorError as exc:
            log.warning("download failed for %s: %s", candidate.key, exc)
            return candidate, None

    results: list[tuple[Candidate, bytes | None]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for item in pool.map(fetch, candidates):
            results.append(item)
    return results


def collect(
    store: DatasetStore,
    targets: Sequence[Target],
    *,
    http: HttpClient | None = None,
    per_label: int = DEFAULT_TARGET,
    max_new: int | None = None,
    workers: int = DEFAULT_WORKERS,
    labels: Sequence[str] | None = None,
    providers: Sequence[str] = ("inat", "commons"),
    dry_run: bool = False,
    retry_exhausted: bool = False,
    allow_unverified_commons: bool = True,
    today: str | None = None,
) -> CollectResult:
    http = http or HttpClient()
    provider_map: dict[str, object] = {
        "inat": INaturalistProvider(http),
        "commons": CommonsProvider(http),
    }
    active_providers = [provider_map[name] for name in providers if name in provider_map]
    if not active_providers:
        raise CollectorError("no usable providers selected")

    store.load()
    state = store.load_state()
    label_state = state.setdefault("labels", {})
    date_accessed = today or dt.date.today().isoformat()
    result = CollectResult()
    wanted = {label.lower() for label in labels} if labels else None

    for target in targets:
        if wanted is not None and target.label.lower() not in wanted:
            continue
        have = store.count(target.label)
        need = per_label - have
        if need <= 0:
            continue
        if max_new is not None:
            remaining = max_new - result.added
            if remaining <= 0:
                log.info("max-new reached; stopping")
                break
            need = min(need, remaining)

        entry = label_state.setdefault(target.label, {})
        if entry.get("exhausted") and not retry_exhausted:
            log.info("%s marked exhausted; use --retry-exhausted to retry", target.label)
            continue

        log.info("collecting %s: have %d, need %d", target.display, have, need)
        candidates = iter_candidates(active_providers, target, limit=max(need * 8, 12))
        # Prefer confirmed iNaturalist records at research grade.
        candidates.sort(key=lambda c: (not c.taxon_confirmed, c.quality_grade != "research"))
        # Never re-download a photograph that is already in the dataset.
        existing_keys = {
            (row.data.get("source", ""), row.data.get("source_record_id", ""))
            for row in store.rows
        }
        candidates = [
            c for c in candidates if (c.provider, c.record_id) not in existing_keys
        ]
        if not candidates:
            log.info("%s: only already-collected candidates found", target.label)
        if not allow_unverified_commons:
            candidates = [c for c in candidates if c.taxon_confirmed]
        if dry_run:
            log.info(
                "dry-run %s: %d candidate(s): %s",
                target.display,
                len(candidates),
                ", ".join(c.key for c in candidates[:need]),
            )
            result.attempts += len(candidates)
            continue

        if not candidates:
            entry["exhausted"] = True
            entry["reason"] = "no candidates after filters"
            store.save_state(state)
            log.warning("no usable candidates for %s", target.display)
            continue

        downloaded = fetch_candidates(http, candidates, workers=workers)
        added_here = 0
        new_rows: list[Row] = []
        for candidate, raw in downloaded:
            if added_here >= need:
                break
            result.attempts += 1
            if raw is None:
                result.skipped += 1
                continue
            try:
                jpeg, width, height = process_image(raw)
            except CollectorError as exc:
                log.info("reject %s: %s", candidate.key, exc)
                result.rejected += 1
                continue
            if width < 320 or height < 240:
                log.info("reject %s: processed image too small", candidate.key)
                result.rejected += 1
                continue
            try:
                row = store.add_image(
                    target=target,
                    candidate=candidate,
                    jpeg=jpeg,
                    width=width,
                    height=height,
                    date_accessed=date_accessed,
                )
            except DuplicateImage as exc:
                log.info("skip %s: %s", candidate.key, exc)
                result.duplicates += 1
                continue
            new_rows.append(row)
            added_here += 1
            result.added += 1
            result.per_label[target.label] = result.per_label.get(target.label, 0) + 1

        if new_rows:
            store.save(target, new_rows)
            log.info("saved %d image(s) for %s", len(new_rows), target.label)
        entry["last_attempt"] = dt.datetime.now().isoformat(timespec="seconds")
        entry["added"] = entry.get("added", 0) + added_here
        if added_here < need:
            entry["exhausted"] = True
            entry["reason"] = "candidate pool exhausted"
        else:
            entry.pop("exhausted", None)
            entry.pop("reason", None)
        store.save_state(state)

    if not dry_run:
        store.write_metadata()
        update_progress(store, targets_cache={})
    return result


# ---------------------------------------------------------------------------
# Validation, status, contact sheets, repair
# ---------------------------------------------------------------------------


def validate_dataset(root: Path) -> list[str]:
    issues: list[str] = []
    metadata = root / "metadata.csv"
    if not metadata.exists():
        return [f"missing {metadata}"]
    with metadata.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_fields = [f for f in METADATA_FIELDS if f not in (reader.fieldnames or [])]
        if missing_fields:
            issues.append(f"metadata missing columns: {', '.join(missing_fields)}")
        rows = list(reader)

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_sha: dict[str, str] = {}
    hashes: list[tuple[str, int]] = []
    for row in rows:
        sample_id = row.get("sample_id", "?")
        if sample_id in seen_ids:
            issues.append(f"duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        relative = row.get("relative_path", "")
        if relative in seen_paths:
            issues.append(f"duplicate path: {relative}")
        seen_paths.add(relative)
        path = root / relative
        if not path.exists():
            issues.append(f"missing file: {relative}")
            continue
        if path.stat().st_size != int(row.get("bytes") or -1):
            issues.append(f"byte count mismatch: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row.get("sha256"):
            issues.append(f"sha256 mismatch: {relative}")
        if digest in seen_sha:
            issues.append(f"exact duplicate: {relative} == {seen_sha[digest]}")
        seen_sha[digest] = relative
        for field in ("license", "license_url", "source_page_url", "author", "observed_scientific"):
            if not row.get(field, "").strip():
                issues.append(f"empty {field}: {relative}")
        if parse_license(row.get("license")) is None:
            issues.append(f"unaccepted licence: {relative} ({row.get('license')})")
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
        except (OSError, UnidentifiedImageError):
            issues.append(f"undecodable image: {relative}")
            continue
        if (width, height) != (int(row.get("width") or 0), int(row.get("height") or 0)):
            issues.append(f"dimension mismatch: {relative}")
        if max(width, height) > MAX_EDGE:
            issues.append(f"image larger than {MAX_EDGE}px: {relative}")
        hashes.append((sample_id, dhash(path.read_bytes())))

    for i, (sid_a, hash_a) in enumerate(hashes):
        for sid_b, hash_b in hashes[i + 1 :]:
            if hamming(hash_a, hash_b) <= DHASH_DISTANCE:
                issues.append(f"near-duplicate: {sid_a} ~ {sid_b}")

    checksums = root / "CHECKSUMS.sha256"
    if not checksums.exists():
        issues.append("missing CHECKSUMS.sha256")
    else:
        listed = {}
        for line in checksums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, _, name = line.partition("  ")
            listed[name] = digest
        if len(listed) != len(rows):
            issues.append(f"checksum entries {len(listed)} != metadata rows {len(rows)}")
        for row in rows:
            relative = row.get("relative_path", "")
            if relative in listed and listed[relative] != row.get("sha256"):
                issues.append(f"checksum mismatch: {relative}")
    return issues


def status_report(store: DatasetStore, targets: Sequence[Target]) -> str:
    store.load()
    store.ensure_layout()
    state = store.load_state().get("labels", {})
    lines = [
        f"Dataset: {store.root}",
        f"Total images: {len(store.rows)}",
        "",
        "| Label | Images | State |",
        "|---|---:|---|",
    ]
    complete = 0
    for target in targets:
        count = store.count(target.label)
        complete += count >= DEFAULT_TARGET
        entry = state.get(target.label, {})
        if count >= DEFAULT_TARGET:
            state_text = "complete"
        elif entry.get("exhausted"):
            state_text = f"short ({entry.get('reason', 'exhausted')})"
        else:
            state_text = "pending"
        lines.append(f"| {target.label} | {count} | {state_text} |")
    lines.insert(2, f"Labels complete: {complete}/{len(targets)}")
    return "\n".join(lines)


def make_contact_sheets(
    root: Path,
    targets: Sequence[Target],
    out_dir: Path,
    *,
    labels: Sequence[str] | None = None,
) -> list[Path]:
    wanted = {label.lower() for label in labels} if labels else None
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    thumb_w, thumb_h, caption = 400, 300, 34
    for target in targets:
        if wanted is not None and target.label.lower() not in wanted:
            continue
        folder = root / target.label
        if not folder.exists():
            continue
        files = sorted(folder.glob("*.jpg"))
        if not files:
            continue
        cols = 4
        rows = (len(files) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + caption)), (24, 24, 24))
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(files):
            try:
                with Image.open(path) as image:
                    thumb = image.convert("RGB")
                    thumb.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
            except (OSError, UnidentifiedImageError):
                continue
            x = (index % cols) * thumb_w
            y = (index // cols) * (thumb_h + caption)
            sheet.paste(thumb, (x + (thumb_w - thumb.width) // 2, y + caption))
            draw.text((x + 6, y + 8), path.stem[:56], fill=(255, 255, 255))
        destination = out_dir / f"{slugify(target.label)}.jpg"
        sheet.save(destination, "JPEG", quality=88)
        produced.append(destination)
    return produced


def repair_metadata(store: DatasetStore, *, apply: bool = False) -> list[str]:
    """Drop rows whose files are missing and rebuild derived files."""
    store.load()
    kept: list[Row] = []
    dropped: list[str] = []
    for row in store.rows:
        if (store.root / row.path).exists():
            kept.append(row)
        else:
            dropped.append(row.sample_id)
    if apply and dropped:
        store.rows = kept
        store.by_label = {}
        for row in kept:
            store.by_label.setdefault(row.label, []).append(row)
        store.write_metadata()
        store.rebuild_checksums()
        update_progress(store, targets_cache={})
    return dropped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def configure_logging(store: DatasetStore, verbose: bool) -> None:
    store.ensure_layout()
    handler = logging.FileHandler(store.error_log, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.handlers.clear()
    log.addHandler(stream)
    log.addHandler(handler)
    log.propagate = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collect_fish_samples.py",
        description="Collect licensed fish photographs into SAMPLES FISH/.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    collect_parser = sub.add_parser("collect", help="download missing images")
    collect_parser.add_argument("--dry-run", action="store_true")
    collect_parser.add_argument("--max-new", type=int, default=None)
    collect_parser.add_argument("--per-label", type=int, default=DEFAULT_TARGET)
    collect_parser.add_argument("--labels", default="", help="comma separated labels")
    collect_parser.add_argument("--providers", default="inat,commons")
    collect_parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    collect_parser.add_argument("--retry-exhausted", action="store_true")
    collect_parser.add_argument(
        "--no-unverified-commons",
        action="store_true",
        help="require taxonomic confirmation (iNaturalist only)",
    )

    sub.add_parser("status", help="show per-label counts")
    sub.add_parser("validate", help="verify images, hashes and metadata")

    sheet_parser = sub.add_parser("contact-sheets", help="render review sheets")
    sheet_parser.add_argument("--out", type=Path, default=None)
    sheet_parser.add_argument("--labels", default="")

    repair_parser = sub.add_parser("repair", help="drop rows for missing files")
    repair_parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "collect"
    if args.command is None:
        args.dry_run = False
        args.max_new = None
        args.per_label = DEFAULT_TARGET
        args.labels = ""
        args.providers = "inat,commons"
        args.workers = DEFAULT_WORKERS
        args.retry_exhausted = False
        args.no_unverified_commons = False

    store = DatasetStore(args.dataset)
    store.ensure_layout()
    configure_logging(store, args.verbose)
    targets = load_targets()

    if command == "status":
        print(status_report(store, targets))
        return 0

    if command == "validate":
        store.load()
        issues = validate_dataset(store.root)
        if issues:
            print(f"{len(issues)} issue(s):")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        print(f"OK: {len(store.rows)} images validated")
        return 0

    if command == "contact-sheets":
        labels = split_labels(args.labels)
        out_dir = args.out or store.root / "_contact_sheets"
        produced = make_contact_sheets(store.root, targets, out_dir, labels=labels)
        print(f"wrote {len(produced)} sheet(s) to {out_dir}")
        return 0

    if command == "repair":
        dropped = repair_metadata(store, apply=args.apply)
        verb = "removed" if args.apply else "would remove"
        print(f"{verb} {len(dropped)} row(s) with missing files")
        for sample_id in dropped:
            print(f"  - {sample_id}")
        return 0

    if command == "collect":
        labels = split_labels(args.labels)
        providers = [p.strip() for p in args.providers.split(",") if p.strip()]
        try:
            with store.lock():
                result = collect(
                    store,
                    targets,
                    per_label=args.per_label,
                    max_new=args.max_new,
                    workers=args.workers,
                    labels=labels or None,
                    providers=providers,
                    dry_run=args.dry_run,
                    retry_exhausted=args.retry_exhausted,
                    allow_unverified_commons=not args.no_unverified_commons,
                )
        except LockHeld as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(
            f"added={result.added} attempts={result.attempts} "
            f"duplicates={result.duplicates} rejected={result.rejected} "
            f"download_failures={result.skipped}"
        )
        if result.per_label:
            for label, count in sorted(result.per_label.items()):
                print(f"  {label}: +{count}")
        return 0

    parser.error(f"unknown command: {command}")
    return 2


def split_labels(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
