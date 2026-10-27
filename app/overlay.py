"""Video overlay rendering and the camera-offline placeholder.

The overlay intentionally stays minimal: bounding boxes for the subject that
produced the current decision, a small live HUD, and nothing that could be
mistaken for a prerecorded feed.  All text is drawn from live state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .decision import FinalResult

# BGR colours.
_GREEN = (129, 211, 52)
_BLUE = (250, 164, 96)
_AMBER = (79, 179, 245)
_WHITE = (244, 247, 250)
_MUTED = (170, 180, 190)
_DARK = (24, 20, 18)
_BG = (21, 17, 15)
_GRID = (40, 35, 32)

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class OverlayMeta:
    width: int = 0
    height: int = 0
    camera_fps: float = 0.0
    inference_ms: float = 0.0
    device: str = ""
    status: str = ""


def _accent(result: FinalResult) -> tuple[int, int, int]:
    if result.fish:
        return _GREEN
    if result.type_of_fish == "Human":
        return _BLUE
    return _AMBER


def _label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float = 0.52,
    thickness: int = 1,
) -> tuple[int, int, int, int]:
    font_scale = scale
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    pad_x, pad_y = 9, 7
    x, y = origin
    h, w = image.shape[:2]
    x = max(2, min(x, w - text_w - 2 * pad_x - 2))
    y = max(2, min(y, h - text_h - 2 * pad_y - baseline - 2))
    box = (x, y, x + text_w + 2 * pad_x, y + text_h + 2 * pad_y + baseline)
    layer = image.copy()
    cv2.rectangle(layer, box[:2], box[2:], _DARK, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.82, image, 0.18, 0, image)
    cv2.rectangle(image, box[:2], box[2:], color, 1, cv2.LINE_AA)
    cv2.putText(
        image,
        text,
        (x + pad_x, y + pad_y + text_h),
        FONT,
        font_scale,
        _WHITE,
        thickness,
        cv2.LINE_AA,
    )
    return box


def _corner_box(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    h, w = image.shape[:2]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return
    arm = int(max(12, min(w, h) * 0.035))
    arm = min(arm, (x2 - x1) // 3 + 1, (y2 - y1) // 3 + 1)
    # Faint full rectangle keeps the subject readable on busy backgrounds.
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    for sx, sy, dx, dy in (
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ):
        cv2.line(image, (sx, sy), (sx + dx * arm, sy), color, thickness, cv2.LINE_AA)
        cv2.line(image, (sx, sy), (sx, sy + dy * arm), color, thickness, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    result: FinalResult | None,
    meta: OverlayMeta | None = None,
) -> np.ndarray:
    """Return a copy of ``frame`` with boxes and a compact live HUD."""
    if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("Cannot draw an overlay on an empty frame")
    out = frame.copy()
    h, w = out.shape[:2]
    meta = meta or OverlayMeta()

    if result is not None:
        fish_box = result.fish_box
        face_box = result.face
        accent = _accent(result)

        if fish_box is not None:
            x1, y1 = int(fish_box.x1 * w), int(fish_box.y1 * h)
            x2, y2 = int(fish_box.x2 * w), int(fish_box.y2 * h)
            _corner_box(out, x1, y1, x2, y2, _GREEN)
            label = (
                f"{result.type_of_fish.upper()}  {result.confidence * 100:.0f}%"
                if result.fish
                else "FISH CANDIDATE"
            )
            _label(out, label, (x1, max(2, y1 - 34)), _GREEN)

        if face_box is not None:
            x1, y1 = int(face_box.x1 * w), int(face_box.y1 * h)
            x2, y2 = int(face_box.x2 * w), int(face_box.y2 * h)
            _corner_box(out, x1, y1, x2, y2, _BLUE)
            label = (
                "HUMAN  "
                f"{result.confidence * 100:.0f}%"
                if result.type_of_fish == "Human"
                else "FACE (BACKGROUND)"
            )
            _label(out, label, (x1, max(2, y1 - 34)), _BLUE)

        # Decision chip in the lower-left: the single source of truth.
        chip = result.display["line2"]
        if result.fish:
            chip = f"FISH  |  {result.type_of_fish}"
        elif result.type_of_fish == "Human":
            chip = "HUMAN"
        else:
            chip = "NOT FISH"
        (chip_w, chip_h), chip_base = cv2.getTextSize(chip, FONT, 0.58, 1)
        chip_bottom = h - 42
        chip_box = (
            18,
            chip_bottom - chip_h - chip_base - 16,
            18 + chip_w + 24,
            chip_bottom,
        )
        layer = out.copy()
        cv2.rectangle(layer, chip_box[:2], chip_box[2:], _DARK, -1, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.78, out, 0.22, 0, out)
        cv2.rectangle(out, chip_box[:2], chip_box[2:], accent, 1, cv2.LINE_AA)
        cv2.putText(
            out,
            chip,
            (chip_box[0] + 12, chip_box[3] - chip_base - 8),
            FONT,
            0.58,
            _WHITE,
            1,
            cv2.LINE_AA,
        )

    # HUD ------------------------------------------------------------------
    dot = (34, 34 + 9)
    cv2.circle(out, dot, 4, _GREEN if meta.status == "running" else _AMBER, -1, cv2.LINE_AA)
    live_text = "LIVE" if meta.status == "running" else "OFFLINE"
    cv2.putText(out, live_text, (48, 39), FONT, 0.48, _WHITE, 1, cv2.LINE_AA)

    details: list[str] = []
    if meta.width and meta.height:
        details.append(f"{meta.width}x{meta.height}")
    if meta.camera_fps > 0:
        details.append(f"{meta.camera_fps:.0f} fps")
    if details:
        cv2.putText(
            out,
            "  ".join(details),
            (100, 39),
            FONT_MONO,
            0.43,
            _MUTED,
            1,
            cv2.LINE_AA,
        )

    right: list[str] = []
    if meta.device:
        right.append(meta.device.upper())
    if meta.inference_ms > 0:
        right.append(f"{meta.inference_ms:.0f} ms")
    if right:
        text = "  |  ".join(right)
        (tw, _), _ = cv2.getTextSize(text, FONT_MONO, 0.43, 1)
        cv2.putText(
            out,
            text,
            (w - tw - 20, 39),
            FONT_MONO,
            0.43,
            _MUTED,
            1,
            cv2.LINE_AA,
        )

    stamp = time.strftime("%H:%M:%S")
    (sw, _), _ = cv2.getTextSize(stamp, FONT_MONO, 0.40, 1)
    cv2.putText(
        out,
        stamp,
        (w - sw - 20, h - 20),
        FONT_MONO,
        0.40,
        (190, 195, 200),
        1,
        cv2.LINE_AA,
    )
    return out


def make_placeholder(
    width: int = 1280,
    height: int = 720,
    message: str = "CAMERA OFFLINE",
    detail: str = "Press Start Camera to begin live recognition",
) -> np.ndarray:
    """A clearly-labelled placeholder; never a synthetic or recorded feed."""
    width = max(320, int(width))
    height = max(180, int(height))
    image = np.full((height, width, 3), _BG, dtype=np.uint8)

    # Subtle grid.
    step = max(40, width // 24)
    for x in range(0, width, step):
        cv2.line(image, (x, 0), (x, height), _GRID, 1)
    for y in range(0, height, step):
        cv2.line(image, (0, y), (width, y), _GRID, 1)

    cx, cy = width // 2, height // 2
    # Camera outline.
    scale = max(0.7, min(1.6, width / 1280.0))
    bw, bh = int(112 * scale), int(82 * scale)
    x1, y1 = cx - bw // 2, cy - bh // 2 - int(22 * scale)
    cv2.rectangle(image, (x1, y1), (x1 + bw, y1 + bh), (86, 92, 100), 2, cv2.LINE_AA)
    cv2.circle(image, (cx, y1 + bh // 2), int(24 * scale), (86, 92, 100), 2, cv2.LINE_AA)
    cv2.circle(image, (cx, y1 + bh // 2), int(10 * scale), (86, 92, 100), 2, cv2.LINE_AA)
    cv2.line(
        image,
        (x1 + int(16 * scale), y1 - int(16 * scale)),
        (x1 + int(46 * scale), y1 - int(16 * scale)),
        (86, 92, 100),
        max(2, int(3 * scale)),
        cv2.LINE_AA,
    )
    # Slash to make the offline state unambiguous.
    cv2.line(
        image,
        (x1 - int(10 * scale), y1 + bh + int(10 * scale)),
        (x1 + bw + int(10 * scale), y1 - int(10 * scale)),
        _AMBER,
        max(2, int(3 * scale)),
        cv2.LINE_AA,
    )

    font_scale = max(0.7, min(1.25, width / 1280.0))
    (tw, th), _ = cv2.getTextSize(message, FONT, font_scale, 2)
    cv2.putText(
        image,
        message,
        (cx - tw // 2, cy + int(70 * scale)),
        FONT,
        font_scale,
        _WHITE,
        2,
        cv2.LINE_AA,
    )
    sub_scale = max(0.42, font_scale * 0.44)
    (dw, _), _ = cv2.getTextSize(detail, FONT_MONO, sub_scale, 1)
    cv2.putText(
        image,
        detail,
        (cx - dw // 2, cy + int(104 * scale)),
        FONT_MONO,
        sub_scale,
        _MUTED,
        1,
        cv2.LINE_AA,
    )
    return image


def encode_jpeg(image: np.ndarray, quality: int = 82) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(max(40, min(96, quality)))]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()
