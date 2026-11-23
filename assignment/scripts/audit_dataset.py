#!/usr/bin/env python3
"""Phase 1 audit for assignment S79244.

Read-only with respect to SAMPLES FISH/. Produces integrity, quality,
metadata, duplicate and class-mapping reports plus contact sheets under
``assignment/reports/``. The class mapping is read from the authoritative
``assignment/config/mapping.yaml`` used by the training pipeline.

Usage:
    python assignment/scripts/audit_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageStat

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.collect_fish_samples import dhash, hamming, parse_license  # noqa: E402

DEFAULT_DATASET = REPO / "SAMPLES FISH"
DEFAULT_MAPPING = REPO / "assignment" / "config" / "mapping.yaml"
DEFAULT_OUT = REPO / "assignment" / "reports"
DHASH_THRESHOLD = 6
MIN_EDGE = 400
BLUR_THRESHOLD = 8.0
MEAN_LUMA_RANGE = (25, 245)


def load_labels() -> list[str]:
    spec = importlib.util.spec_from_file_location("_assignment_species", REPO / "app" / "species.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return [s.name for s in module.SPECIES]


def load_mapping(path: Path) -> tuple[dict, dict[str, str], list[dict]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    label_to_class: dict[str, str] = {}
    classes: list[dict] = []
    for category in data["categories"]:
        for label in category["labels"]:
            label_to_class[label] = category["id"]
        classes.append(
            {
                "name": category["id"],
                "title": category["name"],
                "criterion": category["criterion"],
                "labels": len(category["labels"]),
                "expected_images": 4 * len(category["labels"]),
            }
        )
    return data, label_to_class, classes


def quality_metrics(path: Path) -> dict:
    with Image.open(path) as image:
        image.load()
        oriented = ImageOps.exif_transpose(image)
        width, height = oriented.size
        gray = oriented.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        sharpness = float(ImageStat.Stat(edges).stddev[0])
        luma = float(ImageStat.Stat(gray).mean[0])
    ratio = max(width, height) / max(1, min(width, height))
    flags = []
    if min(width, height) < MIN_EDGE:
        flags.append("low_resolution")
    if sharpness < BLUR_THRESHOLD:
        flags.append("low_detail")
    if not MEAN_LUMA_RANGE[0] <= luma <= MEAN_LUMA_RANGE[1]:
        flags.append("exposure")
    if ratio > 4.0:
        flags.append("extreme_aspect")
    return {
        "width": width,
        "height": height,
        "sharpness": round(sharpness, 2),
        "mean_luma": round(luma, 2),
        "aspect": round(ratio, 3),
        "quality_flags": flags,
    }


def audit(dataset: Path, mapping_path: Path, out: Path) -> dict:
    mapping, label_to_class, classes = load_mapping(mapping_path)
    labels = load_labels()
    partition_issues = []
    for label in labels:
        if label not in label_to_class:
            partition_issues.append(f"unassigned: {label}")
    partition_issues.extend(
        f"unknown label: {label}" for label in sorted(set(label_to_class) - set(labels))
    )

    metadata_path = dataset / "metadata.csv"
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    records: list[dict] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    perceptual: list[tuple[str, int]] = []
    checksums = {}
    checksum_path = dataset / "CHECKSUMS.sha256"
    if checksum_path.exists():
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            digest, sep, name = line.partition("  ")
            if sep:
                checksums[name] = digest

    for row in rows:
        path = dataset / row["relative_path"]
        issues = []
        record = {
            "sample_id": row["sample_id"],
            "target_label": row["target_label"],
            "class": label_to_class.get(row["target_label"], "UNASSIGNED"),
            "path": row["relative_path"],
            "source": row["source"],
            "source_id": row["source_record_id"],
            "license": row["license"],
            "human_status": row["human_status"],
            "automated_status": row["automated_status"],
            "issues": issues,
        }
        if not path.exists():
            issues.append("missing_file")
        else:
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            hashes[digest].append(row["sample_id"])
            record["sha256"] = digest
            if digest != row.get("sha256"):
                issues.append("sha256_mismatch")
            if checksums.get(row["relative_path"]) not in (None, digest):
                issues.append("checksum_file_mismatch")
            if len(payload) != int(row.get("bytes") or -1):
                issues.append("byte_count_mismatch")
            try:
                metrics = quality_metrics(path)
                record.update(metrics)
                issues.extend(metrics["quality_flags"])
                perceptual.append((row["sample_id"], dhash(payload)))
            except Exception as exc:  # noqa: BLE001 - audit must not abort
                issues.append(f"decode_failed: {type(exc).__name__}: {exc}")
        for field in (
            "source_page_url",
            "original_image_url",
            "license_url",
            "author",
            "attribution",
            "observed_scientific",
        ):
            if not row.get(field, "").strip():
                issues.append(f"missing_{field}")
        if parse_license(row.get("license")) is None:
            issues.append("license_unacceptable")
        if row.get("human_status") != "verified":
            issues.append("human_review_pending")
        if row.get("automated_status") == "needs_review":
            issues.append("automated_review_needed")
        records.append(record)

    exact_duplicates = [ids for ids in hashes.values() if len(ids) > 1]
    near_duplicates = []
    for i, (sid_a, hash_a) in enumerate(perceptual):
        for sid_b, hash_b in perceptual[i + 1 :]:
            distance = hamming(hash_a, hash_b)
            if distance <= DHASH_THRESHOLD:
                near_duplicates.append({"a": sid_a, "b": sid_b, "dhash_distance": distance})

    counts_by_class = Counter(r["class"] for r in records)
    eligible = [
        r
        for r in records
        if not any(
            i in r["issues"]
            for i in (
                "missing_file",
                "sha256_mismatch",
                "checksum_file_mismatch",
                "byte_count_mismatch",
                "license_unacceptable",
                "human_review_pending",
                "automated_review_needed",
                "low_resolution",
                "low_detail",
                "exposure",
                "extreme_aspect",
            )
        )
        and not any(i.startswith(("decode_failed", "missing_")) for i in r["issues"])
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "mapping": str(mapping_path),
        "mapping_status": mapping.get("status_note", "provisional — NOT approved"),
        "metadata_rows": len(rows),
        "files_on_disk": len(list(dataset.rglob("*.jpg"))),
        "partition_issues": partition_issues,
        "integrity": {
            "missing_files": sum("missing_file" in r["issues"] for r in records),
            "decode_failures": sum(any(i.startswith("decode_failed") for i in r["issues"]) for r in records),
            "sha256_mismatches": sum("sha256_mismatch" in r["issues"] for r in records),
            "exact_duplicate_groups": exact_duplicates,
            "near_duplicate_pairs": near_duplicates,
        },
        "quality": {
            flag: sum(flag in r["issues"] for r in records)
            for flag in ("low_resolution", "low_detail", "exposure", "extreme_aspect")
        },
        "metadata": {
            "license_unacceptable": sum("license_unacceptable" in r["issues"] for r in records),
            "incomplete_fields": sum(any(i.startswith("missing_") for i in r["issues"]) for r in records),
            "human_review_pending": sum("human_review_pending" in r["issues"] for r in records),
            "automated_review_needed": sum("automated_review_needed" in r["issues"] for r in records),
        },
        "classes": [
            {
                "name": c["name"],
                "title": c["title"],
                "criterion": c["criterion"],
                "labels": c["labels"],
                "images": counts_by_class[c["name"]],
                "expected_images": c["expected_images"],
            }
            for c in classes
        ],
        "eligible_without_review": len(eligible),
        "stop_conditions": [],
        "records": records,
    }
    if partition_issues:
        report["stop_conditions"].append("class mapping does not partition the 80 labels")
    if report["eligible_without_review"] < 300:
        report["stop_conditions"].append("fewer than 300 human-verified eligible images")
    if report["integrity"]["missing_files"] or report["integrity"]["decode_failures"]:
        report["stop_conditions"].append("corrupt or missing images")

    out.mkdir(parents=True, exist_ok=True)
    (out / "phase1_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, out / "phase1_audit.md")
    write_contact_sheets(dataset, classes, records, out / "contact_sheets")
    summary = {k: v for k, v in report.items() if k != "records"}
    return summary


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Phase 1 — dataset audit (S79244)",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Mapping: `{report['mapping']}` ({report['mapping_status']})",
        "",
        f"- Metadata rows: **{report['metadata_rows']}**",
        f"- JPG files on disk (includes generated review sheets): {report['files_on_disk']}",
        f"- Eligible under strict human-review gate: **{report['eligible_without_review']}**",
        "",
        "## Integrity and duplicates",
        "",
        f"- Missing files: {report['integrity']['missing_files']}",
        f"- Decode failures: {report['integrity']['decode_failures']}",
        f"- SHA-256 mismatches: {report['integrity']['sha256_mismatches']}",
        f"- Exact duplicate groups: {len(report['integrity']['exact_duplicate_groups'])}",
        f"- Near-duplicate pairs (dHash <= {DHASH_THRESHOLD}): {len(report['integrity']['near_duplicate_pairs'])}",
        "",
        "## Quality and metadata screens",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in report["quality"].items())
    lines.append("")
    lines.extend(f"- {key}: {value}" for key, value in report["metadata"].items())
    lines.extend(["", "## Proposed five classes (provisional — approval required)", "", "| Class | Labels | Images | Target |", "|---|---:|---:|---:|"])
    lines.extend(
        f"| {c['name']} | {c['labels']} | {c['images']} | {c['expected_images']} |"
        for c in report["classes"]
    )
    lines.extend(["", "## Stop conditions", ""])
    lines.extend(f"- {item}" for item in report["stop_conditions"]) or lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_contact_sheets(dataset: Path, classes: list[dict], records: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_class[record["class"]].append(record)
    thumb_w, thumb_h, caption, cols = 220, 165, 18, 8
    for index, cls in enumerate(classes, start=1):
        members = by_class[cls["name"]]
        rows_needed = (len(members) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb_w, rows_needed * (thumb_h + caption)), (24, 24, 24))
        draw = ImageDraw.Draw(sheet)
        for i, record in enumerate(members):
            path = dataset / record["path"]
            if not path.exists():
                continue
            try:
                with Image.open(path) as image:
                    thumb = image.convert("RGB")
                    thumb.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
            except Exception:  # noqa: BLE001
                continue
            x = (i % cols) * thumb_w
            y = (i // cols) * (thumb_h + caption)
            sheet.paste(thumb, (x + (thumb_w - thumb.width) // 2, y + caption))
            draw.text((x + 4, y + 3), f"{record['sample_id']} | {record['source_id']}"[:34], fill=(255, 255, 255))
        destination = out / f"class_{index}.jpg"
        sheet.save(destination, "JPEG", quality=88)
        print(f"sheet: {destination.name} ({len(members)} images)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = audit(args.dataset, args.mapping, args.out)
    print(json.dumps(summary, indent=2))
    return 1 if summary["stop_conditions"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
