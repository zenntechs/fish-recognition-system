"""Command-line interface for the isolated training pipeline.

Commands:
  prepare  Build and validate the dataset manifest, split 70/15/15, save artifacts.
  train    Train a ResNet-18 five-class classifier with early stopping.
  evaluate Evaluate a saved checkpoint on the held-out test set.
  review   Human-review workflow (list, decide, summary, contact-sheets).

All commands run under ``assignment/`` and never touch ``SAMPLES FISH/``,
the collector, or the existing webcam application.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assignment.data.manifest import (  # noqa: E402
    load_manifest,
    manifest_summary,
    save_manifest_json,
    validate_manifest,
    METADATA_CSV,
    MAPPING_YAML,
)
from assignment.data.review import ReviewManifest  # noqa: E402
from assignment.data.split import split_entries, save_split  # noqa: E402
from assignment.data.dataset import build_datasets  # noqa: E402
from assignment.train import run_training, save_run, TrainConfig  # noqa: E402
from assignment.evaluate import (  # noqa: E402
    evaluate_model,
    write_results,
    plot_confusion_matrix,
    plot_training_curves,
)
from assignment.model.backbone import ModelConfig, load_checkpoint  # noqa: E402

DEFAULT_CATEGORIES = [
    "cartilaginous_fishes",
    "freshwater_bony_fishes",
    "marine_reef_bony_fishes",
    "marine_openwater_bottom_bony_fishes",
    "diadromous_euryhaline_bony_fishes",
]


def _load_mapping_categories() -> list[str]:
    import yaml

    with MAPPING_YAML.open("r", encoding="utf-8") as fh:
        mapping = yaml.safe_load(fh)
    return [c["id"] for c in mapping.get("categories", [])]


def _load_review_manifest(review_manifest_path: str | None) -> ReviewManifest | None:
    if not review_manifest_path:
        return None
    p = Path(review_manifest_path)
    if not p.is_file():
        return None
    return ReviewManifest.load(p)


MIN_ELIGIBLE_IMAGES = 300
MIN_CLASSES = 3
MAX_CLASS_RATIO = 2.0


def _eligibility_gate_problems(entries) -> list[str]:
    """Expedited-mode gates; strict mode is deliberately unaffected."""
    from collections import Counter

    eligible = [e for e in entries if e.eligible]
    problems = []
    if len(eligible) < MIN_ELIGIBLE_IMAGES:
        problems.append(
            f"only {len(eligible)} eligible images; need >= {MIN_ELIGIBLE_IMAGES} "
            f"(shortfall {MIN_ELIGIBLE_IMAGES - len(eligible)})"
        )
    counts = Counter(e.category for e in eligible if e.category)
    if len(counts) < MIN_CLASSES:
        problems.append(f"only {len(counts)} eligible classes; need >= {MIN_CLASSES}")
    if counts and max(counts.values()) / min(counts.values()) > MAX_CLASS_RATIO:
        problems.append(
            f"class sizes not reasonably balanced: {dict(sorted(counts.items()))} "
            f"(max/min={max(counts.values()) / min(counts.values()):.2f})"
        )
    return problems


def _write_screening_audit(entries, out_dir: Path, review_policy: str) -> Path:
    """Audit trail: auto-screened vs genuinely manually reviewed images."""
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in entries:
        reason = e.eligibility_reason or ""
        if reason.startswith("human_review=approved"):
            basis = "manual_review_approved"
        elif reason.startswith("automated_screening=passed"):
            basis = "automated_screening_provisional"
        else:
            basis = "ineligible"
        rows.append({
            "sample_id": e.sample_id,
            "relative_path": e.relative_path,
            "basis": basis,
            "eligible": e.eligible,
            "automated_status_source": e.automated_status,
            "human_status_source": e.human_status,
            "human_status_changed": False,
            "reason": reason,
        })
    path = out_dir / "screening_audit.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "review_policy": review_policy,
        "total": len(rows),
        "automated_screening_provisional": sum(r["basis"] == "automated_screening_provisional" for r in rows),
        "manual_review_approved": sum(r["basis"] == "manual_review_approved" for r in rows),
        "ineligible": sum(r["basis"] == "ineligible" for r in rows),
        "human_status_modified": False,
        "review_decisions_created": 0,
    }
    (out_dir / "screening_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return path


def cmd_prepare(args: argparse.Namespace) -> int:
    categories = _load_mapping_categories()
    verified = frozenset(args.verified_human_statuses) if args.verified_human_statuses else frozenset()
    review = _load_review_manifest(args.review_manifest)

    entries = load_manifest(
        metadata_csv=METADATA_CSV,
        mapping_path=MAPPING_YAML,
        verified_human_statuses=verified,
        treat_needs_review_as_ineligible=not args.include_needs_review,
        review_manifest=review,
        review_policy=args.review_policy,
    )

    problems = validate_manifest(entries)
    if problems:
        print("Manifest validation problems:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_manifest_json(entries, out_dir / "manifest.json")

    if args.review_policy == "automated-screening":
        audit_path = _write_screening_audit(entries, out_dir, args.review_policy)
        gate_problems = _eligibility_gate_problems(entries)
        if gate_problems:
            print("Expedited preparation stopped:", file=sys.stderr)
            for problem in gate_problems:
                print(f"  - {problem}", file=sys.stderr)
            print(f"Audit trail: {audit_path}", file=sys.stderr)
            print(
                "Add genuine human approvals with `assignment.cli review decide-batch` "
                "and rerun prepare.",
                file=sys.stderr,
            )
            return 1

    split_result = split_entries(entries, seed=args.seed)
    save_split(split_result, out_dir / "split_manifest.json")

    summary = manifest_summary(entries)
    split_summary = split_result.summary()
    (out_dir / "summary.json").write_text(
        json.dumps(
            {"manifest": summary, "split": split_summary}, indent=2
        ),
        encoding="utf-8",
    )

    # Save review summary alongside if a review manifest was used.
    if review is not None:
        (out_dir / "review_summary.json").write_text(
            json.dumps(review.summary(), indent=2), encoding="utf-8"
        )

    print(f"Manifest: {summary['total_images']} images, {summary['eligible_images']} eligible")
    print(f"Labels collected: {summary['labels_collected']}")
    print(f"Per category (all): {summary['per_category_all_labels']}")
    print(f"Per category (eligible): {summary['per_category_eligible']}")
    print(f"Split: train={split_summary['split_counts']['train']} "
          f"val={split_summary['split_counts']['val']} "
          f"test={split_summary['split_counts']['test']}")
    print(f"Group leakage across splits: {split_summary['group_leakage_check']['no_group_in_multiple_splits']}")
    if review is not None:
        print(f"Review decisions: {review.summary()['total_reviewed']} "
              f"(approved={review.summary()['approved']}, "
              f"rejected={review.summary()['rejected']}, "
              f"uncertain={review.summary()['uncertain']})")
    print(f"Artifacts written to {out_dir}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    categories = _load_mapping_categories()
    verified = frozenset(args.verified_human_statuses) if args.verified_human_statuses else frozenset()
    review = _load_review_manifest(args.review_manifest)

    entries = load_manifest(
        metadata_csv=METADATA_CSV,
        mapping_path=MAPPING_YAML,
        verified_human_statuses=verified,
        treat_needs_review_as_ineligible=not args.include_needs_review,
        review_manifest=review,
        review_policy=args.review_policy,
    )
    problems = validate_manifest(entries)
    if problems:
        print("Manifest validation failed; refusing to train.", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    eligible = [e for e in entries if e.eligible]
    if not eligible:
        print("No eligible images for training. Complete human review first.", file=sys.stderr)
        return 1

    if args.review_policy == "automated-screening":
        gate_problems = _eligibility_gate_problems(entries)
        if gate_problems:
            print("Expedited training blocked:", file=sys.stderr)
            for problem in gate_problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        from collections import Counter
        print("Eligibility basis:", dict(Counter(
            "manual_review_approved" if e.eligibility_reason.startswith("human_review=approved")
            else "automated_screening_provisional"
            for e in entries if e.eligible
        )))

    # Check minimum per-class support for train/val/test.
    from collections import Counter
    cat_counts = Counter(e.category for e in eligible if e.category)
    for cat, count in sorted(cat_counts.items()):
        if count < 3:
            print(f"Warning: category {cat!r} has only {count} eligible image(s) "
                  f"– cannot form train/val/test splits.", file=sys.stderr)

    config = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        device=args.device,
        early_stopping_patience=args.patience,
        num_workers=args.num_workers,
        verified_human_statuses=tuple(verified),
        treat_needs_review_as_ineligible=not args.include_needs_review,
        run_dir=Path(args.run_dir),
        image_root=METADATA_CSV.parent,
    )

    results, model, split_result, datasets = run_training(entries, categories, config)
    save_run(results, model, split_result, categories, Path(args.run_dir))

    print(f"Training finished. Epochs run: {results['epochs_run']}")
    print(f"Best epoch: {results['best_epoch']}, best val loss: {results['best_val_loss']:.4f}")
    print(f"Test accuracy: {results['test_accuracy']:.4f}, test loss: {results['test_loss']:.4f}")
    print(f"Stopped early: {results['stopped_early']}")
    print(f"Run artifacts: {args.run_dir}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    categories = _load_mapping_categories()
    checkpoint = Path(args.checkpoint)
    split_manifest = Path(args.split_manifest)

    import json

    split_data = json.loads(split_manifest.read_text(encoding="utf-8"))
    from assignment.data.manifest import ManifestEntry

    entries = load_manifest(
        metadata_csv=METADATA_CSV,
        mapping_path=MAPPING_YAML,
        verified_human_statuses=frozenset(),
        treat_needs_review_as_ineligible=True,
    )
    entry_by_id = {e.sample_id: e for e in entries}

    test_entries = []
    for rec in split_data["splits"]["test"]:
        e = entry_by_id.get(rec["sample_id"])
        if e is None:
            print(f"Warning: test sample {rec['sample_id']} not in manifest", file=sys.stderr)
            continue
        test_entries.append(e)

    from assignment.data.dataset import FishCategoryDataset, eval_transform

    test_dataset = FishCategoryDataset(
        entries=test_entries,
        categories=categories,
        transform=eval_transform(),
        image_root=METADATA_CSV.parent,
    )
    import torch

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_checkpoint(checkpoint, device=device)

    results = evaluate_model(model, test_loader, categories, device=device)
    write_results(results, Path(args.output_dir))
    plot_confusion_matrix(results, Path(args.output_dir) / "confusion_matrix.png")
    if args.history:
        plot_training_curves(json.loads(Path(args.history).read_text(encoding="utf-8")),
                             Path(args.output_dir) / "training_curves.png")

    print(f"Test samples: {results['num_test_samples']}")
    print(f"Overall accuracy: {results['overall_accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Misclassifications: {results['num_misclassifications']}")
    print(f"Results written to {args.output_dir}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Delegate to the review CLI via command-line arguments."""
    # Reconstruct the review CLI args from the parsed args
    review_args = args.review_args
    if not review_args:
        print("No review subcommand provided.", file=sys.stderr)
        print("Usage: python -m assignment.cli review list | decide SAMPLE_ID ...")
        return 2

    # Build a minimal argparse for the review CLI
    from assignment.review.cli import main as review_main
    # The review CLI's main expects sys.argv-like list; we'll call it with the review_args
    # but need to prefix with the review CLI's own program name handling.
    # We can call main with just the review_args since it handles sys.argv internally.
    # To avoid issues, let's call it properly.
    sys.argv[0] = "assignment.review.cli"
    # Replace sys.argv with what the review CLI expects
    # The review CLI's main uses argparse, which reads from sys.argv
    # We need to make sure the first element is the program name.
    # Let's just call the review CLI's main function directly with the args.
    try:
        # Call review_main with empty args first to see structure
        # Actually, let's just pass the review_args properly
        review_main(review_args)
    except SystemExit as e:
        raise
    except Exception as e:
        print(f"Review CLI error: {e}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S79244 Fish Species Detection training pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--review-policy", default="strict", choices=["strict", "automated-screening"],
                       help="strict (default, unchanged) or expedited automated screening")
        p.add_argument("--seed", type=int, default=1337)
        p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--verified-human-statuses", nargs="*", default=[],
                       help="human_status values treated as reviewed (default: none -> nothing eligible)")
        p.add_argument("--include-needs-review", action="store_true",
                       help="Do not exclude automated_status=needs_review images")
        p.add_argument("--review-manifest", default=None,
                       help="Path to assignment-owned review_manifest.json")

    p_prepare = sub.add_parser("prepare", help="Build manifest, validate, split 70/15/15")
    p_prepare.add_argument("--out", default="assignment/artifacts/prepare")
    add_common(p_prepare)

    p_train = sub.add_parser("train", help="Train ResNet-18 with early stopping")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd"])
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--dropout", type=float, default=0.0)
    p_train.add_argument("--freeze-backbone", action="store_true")
    p_train.add_argument("--patience", type=int, default=5)
    p_train.add_argument("--num-workers", type=int, default=2)
    p_train.add_argument("--run-dir", default="assignment/artifacts/run")
    add_common(p_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate checkpoint on held-out test set")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--split-manifest", required=True)
    p_eval.add_argument("--output-dir", default="assignment/artifacts/eval")
    p_eval.add_argument("--history", default=None, help="history.json path for training-curves chart")
    add_common(p_eval)

    p_review = sub.add_parser("review", help="Human-review workflow")
    p_review.add_argument("review_args", nargs=argparse.REMAINDER,
                          help="Arguments passed to assignment.review.cli")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "train":
        return cmd_train(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    if args.command == "review":
        return cmd_review(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())