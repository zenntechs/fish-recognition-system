# Dataset: 320 fish photographs (academic assignment only)

This folder documents the dataset used by the **separate academic four-class
classifier** in `assignment/`. It is not used by the live 80-species webcam app.

## Structure

```text
SAMPLES FISH/
  metadata.csv      320 rows: source, licence, attribution, SHA-256, review state
  <target_label>/   4 images per label, 80 labels (original files, never edited)
  .collector/       collector runtime state (ignored by Git)
```

The directory is **excluded from Git** by `.gitignore`. Do not publish these
third-party photographs without checking each image's redistribution licence and
attribution. The metadata is the source of truth; the collector
(`tools/collect_fish_samples.py`) can recreate the download from the recorded
source URLs subject to network availability and upstream terms.

## Provenance

| Source | Images |
| --- | ---: |
| iNaturalist | 288 |
| Wikimedia Commons | 32 |
| **Total** | **320** |

Every row records `source_page_url`, `original_image_url`, `author`,
`attribution`, `license`, `license_url`, `date_accessed` and `sha256`.

| Licence | Images |
| --- | ---: |
| CC BY 4.0 | 250 |
| CC BY-SA 4.0 | 27 |
| CC0 1.0 | 27 |
| CC BY-SA 3.0 | 5 |
| CC BY 2.0 | 4 |
| CC BY 3.0 | 2 |
| CC0 | 2 |
| Public domain | 2 |
| CC BY-SA 2.0 | 1 |

Share-alike (CC BY-SA) images carry redistribution obligations; CC BY images
require attribution. Verify the exact `license` and `attribution` fields for any
image you intend to publish, and keep the metadata with the images.

## Integrity and review status (verified 2026-11-23)

- 320/320 images decode; 0 missing; 0 SHA-256 mismatches.
- 0 exact duplicates; 0 perceptual near-duplicates.
- 6 quality-flagged images; 28 `automated_status=needs_review`.
- 0 human-verified images; `human_status=pending` throughout.
- Full machine-readable audit: `assignment/reports/phase1_audit.json`.

The historical `Phase_E_audit.md` (276 images / 69 labels) is superseded by the
audit above and is kept only as a record of earlier work.

## Review gate

Training is blocked until a human records genuine approvals in
`assignment/data/review_manifest.json`. That file currently contains one
**test-only** approval from `Test Reviewer`; it is not a genuine human decision
and must be replaced during real review (it was left untouched in cleanup).
Expedited automated screening can mark images *provisionally* eligible but never
changes `human_status` or creates review decisions. See `assignment/README.md`
for the exact commands and audit trail.
