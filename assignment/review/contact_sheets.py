"""Contact-sheet generation for human image review.

Creates PNG contact sheets under ``assignment/data/review/`` showing thumbnails
of candidate images grouped by species and category.  Each sheet includes the
sample ID, target/observed scientific names, source, and automated status so a
human reviewer can make an approve/reject/uncertain decision.

Nothing is written to ``SAMPLES FISH/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from ..data.manifest import ManifestEntry, load_manifest
from ..data.review import ReviewManifest

THUMB_SIZE = 224
PAD = 10
COLS = 4
SHEET_DIR_NAME = "review"


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ):
        p = Path(candidate)
        if p.is_file():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def generate_contact_sheets(
    entries: Sequence[ManifestEntry],
    output_dir: Path | None = None,
    sheets_per_category: bool = True,
) -> list[Path]:
    """Generate contact sheet PNGs for human review.

    Parameters
    ----------
    entries : sequence of ManifestEntry
        Images to include on contact sheets.
    output_dir : path, optional
        Directory for output sheets.  Defaults to
        ``assignment/data/review/``.
    sheets_per_category : bool
        If True, generate one sheet per five-class category.  Otherwise one
        sheet per species label.

    Returns
    -------
    list of Path
        Paths to generated contact sheet PNGs.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data" / SHEET_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    font_name = _load_font(16)
    font_small = _load_font(12)
    font_title = _load_font(20)

    if sheets_per_category:
        groups: dict[str | None, list[ManifestEntry]] = {}
        for e in entries:
            groups.setdefault(e.category, []).append(e)
    else:
        groups = {}
        for e in entries:
            groups.setdefault(e.target_label, []).append(e)

    sheet_paths: list[Path] = []
    for group_name, group_entries in sorted(groups.items(), key=lambda kv: str(kv[0])):
        title = str(group_name) if group_name is not None else "unmapped"
        sheet = _build_sheet(title, group_entries, font_name, font_small, font_title)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)
        out_path = output_dir / f"contact_sheet_{safe_name}.png"
        sheet.save(out_path, format="PNG")
        sheet_paths.append(out_path)
    return sheet_paths


def generate_overview_sheet(
    entries: Sequence[ManifestEntry],
    output_dir: Path | None = None,
    *,
    cols: int = 16,
    thumb: int = 120,
) -> Path:
    """Build ONE compact sheet covering every image, with readable IDs."""
    import math

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data" / SHEET_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    font = _load_font(11)
    rows = math.ceil(len(entries) / cols)
    cell_w, cell_h = thumb + 8, thumb + 26
    sheet = Image.new("RGB", (cols * cell_w + 8, rows * cell_h + 34), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), f"S79244 review overview - {len(entries)} images", fill=(20, 20, 20))
    for i, entry in enumerate(entries):
        x = 4 + (i % cols) * cell_w
        y = 30 + (i // cols) * cell_h
        try:
            with Image.open(entry.absolute_path) as im:
                thumb_im = im.convert("RGB")
                thumb_im.thumbnail((thumb, thumb), Image.LANCZOS)
                cell = Image.new("RGB", (thumb, thumb), (255, 255, 255))
                cell.paste(thumb_im, ((thumb - thumb_im.width) // 2, (thumb - thumb_im.height) // 2))
                sheet.paste(cell, (x, y))
        except Exception:
            draw.rectangle([x, y, x + thumb, y + thumb], outline=(180, 180, 180), fill=(220, 220, 220))
        draw.text((x, y + thumb + 2), entry.sample_id[:20], font=font, fill=(20, 20, 20))
    path = output_dir / "overview_all_images.jpg"
    sheet.save(path, format="JPEG", quality=88)
    return path


def write_review_worksheet(
    entries: Sequence[ManifestEntry],
    review: ReviewManifest,
    output_dir: Path | None = None,
) -> Path:
    """Write a CSV covering all images for efficient human review."""
    import csv

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data" / SHEET_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "review_worksheet.csv"
    fields = [
        "sample_id", "category", "target_label", "relative_path", "automated_status",
        "human_status_source", "current_decision", "decision_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for e in sorted(entries, key=lambda x: (x.category or "", x.sample_id)):
            decision = review.get(e.sample_id)
            writer.writerow({
                "sample_id": e.sample_id,
                "category": e.category or "",
                "target_label": e.target_label,
                "relative_path": e.relative_path,
                "automated_status": e.automated_status,
                "human_status_source": e.human_status,
                "current_decision": decision.decision if decision else "",
                "decision_notes": decision.notes if decision else "",
            })
    return path


def _build_sheet(
    title: str,
    entries: Sequence[ManifestEntry],
    font_name: ImageFont.FreeTypeFont,
    font_small: ImageFont.FreeTypeFont,
    font_title: ImageFont.FreeTypeFont,
) -> Image.Image:
    n = len(entries)
    rows = (n + COLS - 1) // COLS
    cell_w = THUMB_SIZE + 2 * PAD
    cell_h = THUMB_SIZE + 90  # room for text below thumbnail
    sheet_w = COLS * cell_w + (COLS + 1) * PAD
    sheet_h = rows * cell_h + (rows + 1) * PAD + 60

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    # Title
    title_w, _ = _text_size(draw, title, font_title)
    draw.text(((sheet_w - title_w) // 2, PAD // 2), title, font=font_title, fill=(20, 20, 20))

    for idx, entry in enumerate(entries):
        col = idx % COLS
        row = idx // COLS
        x0 = PAD + col * cell_w
        y0 = 60 + PAD + row * cell_h

        # Thumbnail
        thumb_x = x0 + PAD
        thumb_y = y0 + PAD
        try:
            with Image.open(entry.absolute_path) as im:
                im = im.convert("RGB")
                im.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
                # Paste onto a centered white cell
                cell = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), color=(255, 255, 255))
                offset = ((THUMB_SIZE - im.width) // 2, (THUMB_SIZE - im.height) // 2)
                cell.paste(im, offset)
                sheet.paste(cell, (thumb_x, thumb_y))
        except Exception:
            cell = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), color=(220, 220, 220))
            draw.rectangle(
                [thumb_x, thumb_y, thumb_x + THUMB_SIZE, thumb_y + THUMB_SIZE],
                outline=(180, 180, 180),
            )
            sheet.paste(cell, (thumb_x, thumb_y))

        # Text below thumbnail
        text_y = thumb_y + THUMB_SIZE + 6
        lines = [
            entry.sample_id,
            f"target: {entry.target_scientific or entry.target_label}",
        ]
        if entry.observed_scientific and entry.observed_scientific != entry.target_scientific:
            lines.append(f"observed: {entry.observed_scientific}")
        lines.append(f"source: {entry.source or '?'}")
        lines.append(f"auto: {entry.automated_status} | human: {entry.human_status}")

        for line in lines:
            w, _ = _text_size(draw, line, font_small)
            draw.text((thumb_x + max(0, (THUMB_SIZE - w) // 2), text_y), line, font=font_small, fill=(40, 40, 40))
            text_y += 14

    return sheet


def generate_review_index(
    entries: Sequence[ManifestEntry],
    review_manifest: ReviewManifest,
    output_dir: Path | None = None,
) -> Path:
    """Generate a plain-text review index listing every image with its
    current review decision (if any).  Useful as a quick reference."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data" / SHEET_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "review_index.txt"

    lines = ["Fish Image Review Index", "=" * 60, ""]
    decisions = review_manifest.decisions
    for e in sorted(entries, key=lambda x: (x.category or "", x.target_label, x.sample_id)):
        dec = decisions.get(e.sample_id)
        status = dec.decision.upper() if dec else "PENDING"
        lines.append(
            f"{e.sample_id:<32} | {str(e.category):<38} | {e.target_label:<24} | {status}"
        )
        if dec:
            lines.append(f"{'':<34}  reviewer={dec.reviewer} notes={dec.notes}")
    lines.append("")
    lines.append("Legend: PENDING = not yet reviewed by a human.")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
