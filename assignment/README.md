# S79244 — Reproducible fish-group classification

All academic work lives under `assignment/`. It never writes to `SAMPLES FISH/`,
`app/species.py`, the collector, the webcam app, or existing recognition models.

## Status (verified 2026-11-23)

- 320 images / 80 labels / 4 per label; all decode; 0 missing; 0 SHA-256 mismatches;
  0 exact or perceptual near-duplicates; 6 quality-flagged images.
- 28 records are `automated_status=needs_review`; **0 are human-verified**.
- Under the strict review gate **0 images are eligible**. Training is blocked until
  a human records approvals in `assignment/data/review_manifest.json`.
- The 4-class grouping is provisional and needs lecturer approval. This is
  **fish-group classification**, not 80-species detection.

## Files

- `config/mapping.yaml` — authoritative 4-class mapping (provisional)
- `scripts/audit_dataset.py` — read-only integrity/quality/duplicate audit
- `data/`, `model/`, `train.py`, `evaluate.py`, `cli.py` — existing pipeline (reused)
- `review/` — one overview contact sheet + worksheet + review CLI
- `tests/` — 41 passing tests; `report/` — report draft

## Classes (all 320 images, 1.5:1 ratio)

| # | Class | Labels | Images |
|---|---|---:|---:|
| 1 | freshwater_tropical_ornamental | 16 | 64 |
| 2 | freshwater_temperate_game | 20 | 80 |
| 3 | marine_reef_coastal | 20 | 80 |
| 4 | marine_pelagic_demersal | 24 | 96 |

Rules: (1) warm-water freshwater ornamentals; (2) temperate/freshwater-spawning
migratory game fishes; (3) shallow marine reef/coastal/estuarine; (4) fully marine
open-water/bottom, including cartilaginous fishes. Every label is assigned exactly
once. Rationale and ambiguity notes are in `config/mapping.yaml`.

## Model (verified in code)

torchvision ResNet-18, ImageNet-1K pretrained weights, FC head replaced with
`Linear(512 → 4)`, fine-tuning all layers. Input 224×224; train transform
RandomResizedCrop + RandomHorizontalFlip + ColorJitter + ToTensor + ImageNet
normalization; eval transform Resize(256) + CenterCrop(224) + normalization.
Cross-entropy, AdamW (default), fixed seed 1337, early stopping on validation loss
with best-checkpoint saving, CUDA when available.

## Commands

```bash
# Audit (read-only)
.venv/bin/python assignment/scripts/audit_dataset.py

# One overview sheet for all 320 images + review worksheet
.venv/bin/python -m assignment.cli review contact-sheets --single

# Human review: edit assignment/data/review/review_worksheet.csv
# (sample_id,decision[,notes]; decision = approved|rejected|uncertain), then:
.venv/bin/python -m assignment.cli review decide-batch \
  assignment/data/review/review_worksheet.csv --reviewer "Your Name"
.venv/bin/python -m assignment.cli review summary

# Prepare (deterministic stratified group-aware 70/15/15)
.venv/bin/python -m assignment.cli prepare --seed 1337 --device auto \
  --review-manifest assignment/data/review_manifest.json \
  --out assignment/artifacts/prepare

# Train once, then evaluate once on the untouched test set
.venv/bin/python -m assignment.cli train --seed 1337 --device auto \
  --epochs 30 --batch-size 32 --lr 1e-3 --optimizer adamw \
  --weight-decay 1e-4 --patience 5 --run-dir assignment/artifacts/run
.venv/bin/python -m assignment.cli evaluate \
  --checkpoint assignment/artifacts/run/best_model.pt \
  --split-manifest assignment/artifacts/run/split_manifest.json \
  --output-dir assignment/artifacts/eval --device auto

# Tests
.venv/bin/python -m pytest assignment/tests/ -q
```

Reports/figures from evaluate: `metrics.json` (accuracy, per-class accuracy,
precision, recall, F1, support), `confusion_matrix.{csv,png}`,
`training_curves.png`, `per_sample_predictions.csv`, `misclassifications.csv`,
five labelled predictions, `summary.txt`.

## Expedited preparation (opt-in, strict mode unchanged)

```bash
.venv/bin/python -m assignment.cli prepare \
  --review-policy automated-screening --out assignment/artifacts/prepare_expedited
```

`strict` remains the default and is byte-for-byte unchanged. In
`automated-screening` mode, images passing sha256/decodability/quality/resolution/
exact-duplicate/perceptual-duplicate checks are marked **provisionally eligible**;
`human_status` is never modified, no review decision is created, and every
`needs_review` image stays excluded unless a genuine human approval exists in the
review manifest. An audit trail is written to
`assignment/artifacts/prepare_expedited/screening_audit.{csv,json}`.

Current result (verified): **287 eligible**, 28 needs-review + 5 quality-flagged
excluded → **shortfall 13**; the command deliberately stops with exit code 1.
Eligible per class: 52 / 74 / 68 / 93. To proceed, a human must genuinely approve
at least 13 currently excluded images (see `review_worksheet.csv` + overview sheet),
then rerun the same prepare command, then:

```bash
.venv/bin/python -m assignment.cli train --review-policy automated-screening \
  --seed 1337 --device auto --epochs 30 --batch-size 32 --lr 1e-3 \
  --optimizer adamw --weight-decay 1e-4 --patience 5 \
  --run-dir assignment/artifacts/run
.venv/bin/python -m assignment.cli evaluate \
  --checkpoint assignment/artifacts/run/best_model.pt \
  --split-manifest assignment/artifacts/run/split_manifest.json \
  --output-dir assignment/artifacts/eval --device auto
```

Duplicate/observation-group leakage protection and the held-out test set are
unchanged.

Note: `data/review_manifest.json` currently contains one test-only approval
(`Test Reviewer` / `great-white-shark-001`). It is not a genuine human decision
and must be replaced before training; it was left untouched in cleanup.

## Data availability and licensing

`SAMPLES FISH/` is intentionally excluded from Git. It holds third-party
photographs (iNaturalist 288, Wikimedia Commons 32) with per-image licence,
author, attribution and SHA-256 in `SAMPLES FISH/metadata.csv`. Check each
image's redistribution terms before publication; see `docs/DATASET.md` for the
licence summary and reproduction notes. Model checkpoints and generated
training/evaluation outputs under `assignment/artifacts/` are likewise not
committed. No academic checkpoint or evaluation result exists at present.

`Phase_E_audit.md` at the repository root is superseded by
`assignment/reports/phase1_audit.json` (320 images / 80 labels).

## Blocking items

1. 13 genuine human approvals (expedited mode) or full review (strict mode);
   no approvals may be fabricated.
2. Lecturer approval of the four-group mapping / fish-group framing.
