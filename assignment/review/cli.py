"""Human-review command-line tool.

Review decisions are recorded in ``assignment/data/review_manifest.json``
(assignment-owned).  Nothing is ever written to ``SAMPLES FISH/`` and no
image is marked verified without an explicit human decision.

Usage:
    python -m assignment.review.cli list --pending
    python -m assignment.review.cli decide SAMPLE_ID approved --reviewer "Alice"
    python -m assignment.review.cli decide SAMPLE_ID rejected --reviewer "Alice" --notes "blurry"
    python -m assignment.review.cli decide SAMPLE_ID uncertain --reviewer "Alice" --notes "taxonomy unclear"
    python -m assignment.review.cli summary
    python -m assignment.review.cli contact-sheets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assignment.data.manifest import load_manifest, METADATA_CSV, MAPPING_YAML  # noqa: E402
from assignment.data.review import ReviewManifest  # noqa: E402
from assignment.review.contact_sheets import (  # noqa: E402
    generate_contact_sheets,
    generate_overview_sheet,
    generate_review_index,
    write_review_worksheet,
)

REVIEW_DIR = Path(__file__).resolve().parent.parent / "data" / "review"


def cmd_list(args: argparse.Namespace) -> int:
    review = ReviewManifest.load()
    entries = load_manifest(
        metadata_csv=METADATA_CSV,
        mapping_path=MAPPING_YAML,
        verified_human_statuses=frozenset(),
        treat_needs_review_as_ineligible=True,
    )

    pending = [e for e in entries if not review.is_approved(e.sample_id) and not review.is_excluded(e.sample_id)]
    if args.pending:
        entries_to_show = pending
    elif args.all:
        entries_to_show = entries
    else:
        entries_to_show = pending

    print(f"{'Sample ID':<32} | {'Category':<38} | {'Label':<24} | {'Review'}")
    print("-" * 110)
    for e in sorted(entries_to_show, key=lambda x: (x.category or "", x.target_label, x.sample_id)):
        dec = review.get(e.sample_id)
        status = dec.decision.upper() if dec else "pending"
        print(f"{e.sample_id:<32} | {str(e.category or 'unmapped'):<38} | {e.target_label:<24} | {status}")
    print()
    print(f"Total shown: {len(entries_to_show)}")
    print(f"Reviewed: {len(review.decisions)} (approved={review.summary()['approved']}, "
          f"rejected={review.summary()['rejected']}, uncertain={review.summary()['uncertain']})")
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    review = ReviewManifest.load()
    try:
        review.decide(
            sample_id=args.sample_id,
            decision=args.decision,
            reviewer=args.reviewer,
            notes=args.notes,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    review.save()
    print(f"Recorded: {args.sample_id} -> {args.decision} (reviewer={args.reviewer})")
    if args.notes:
        print(f"Notes: {args.notes}")
    return 0


def cmd_decide_batch(args: argparse.Namespace) -> int:
    """Apply an explicit human-provided CSV of decisions (sample_id,decision[,notes])."""
    import csv

    review = ReviewManifest.load()
    applied = skipped = 0
    with open(args.file, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("sample_id") or "").strip()
            decision = (row.get("decision") or "").strip()
            if not sid or not decision:
                continue
            try:
                review.decide(sid, decision, args.reviewer, (row.get("notes") or "").strip())
                applied += 1
            except ValueError as exc:
                print(f"skip {sid}: {exc}")
                skipped += 1
    review.save()
    print(f"Applied {applied} decisions; skipped {skipped}.")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    review = ReviewManifest.load()
    summary = review.summary()
    print("Review manifest summary")
    print("=" * 40)
    print(f"Total reviewed: {summary['total_reviewed']}")
    print(f"  Approved:   {summary['approved']}")
    print(f"  Rejected:   {summary['rejected']}")
    print(f"  Uncertain:  {summary['uncertain']}")
    return 0


def cmd_contact_sheets(args: argparse.Namespace) -> int:
    review = ReviewManifest.load()
    entries = load_manifest(
        metadata_csv=METADATA_CSV,
        mapping_path=MAPPING_YAML,
        verified_human_statuses=frozenset(),
        treat_needs_review_as_ineligible=True,
        review_manifest=review,
    )
    # Only generate sheets for unreviewed images by default
    if args.unreviewed_only:
        entries = [e for e in entries if not review.get(e.sample_id)]
    if getattr(args, "single", False):
        overview = generate_overview_sheet(entries, output_dir=REVIEW_DIR)
        worksheet = write_review_worksheet(entries, review, output_dir=REVIEW_DIR)
        print(f"Overview: {overview}")
        print(f"Worksheet: {worksheet} ({len(entries)} images)")
        return 0
    sheet_paths = generate_contact_sheets(entries, output_dir=REVIEW_DIR, sheets_per_category=True)
    idx_path = generate_review_index(entries, review, output_dir=REVIEW_DIR)
    print(f"Generated {len(sheet_paths)} contact sheets in {REVIEW_DIR}")
    print(f"Review index: {idx_path}")
    for p in sheet_paths:
        print(f"  {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Human review tool for fish dataset")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List images and their review status")
    p_list.add_argument("--pending", action="store_true", help="Show only unreviewed images")
    p_list.add_argument("--all", action="store_true", help="Show all images")

    p_decide = sub.add_parser("decide", help="Record a human review decision")
    p_decide.add_argument("sample_id", help="Sample ID (e.g. clownfish-001)")
    p_decide.add_argument("decision", choices=["approved", "rejected", "uncertain"])
    p_decide.add_argument("--reviewer", required=True, help="Reviewer name")
    p_decide.add_argument("--notes", default="", help="Review notes")

    p_batch = sub.add_parser("decide-batch", help="Apply human-reviewed CSV of decisions")
    p_batch.add_argument("file", help="CSV with columns sample_id,decision[,notes]")
    p_batch.add_argument("--reviewer", required=True, help="Reviewer name")

    sub.add_parser("summary", help="Show review manifest summary")

    p_sheets = sub.add_parser("contact-sheets", help="Generate contact sheets for review")
    p_sheets.add_argument("--unreviewed-only", action="store_true", help="Only sheets for unreviewed images")
    p_sheets.add_argument("--single", action="store_true", help="One overview sheet + full review worksheet")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "decide":
        return cmd_decide(args)
    if args.command == "decide-batch":
        return cmd_decide_batch(args)
    if args.command == "summary":
        return cmd_summary(args)
    if args.command == "contact-sheets":
        return cmd_contact_sheets(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
