"""Application configuration.

All thresholds are deliberately configurable via environment variables so the
classifier can be tuned without touching the decision engine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def _offline_default() -> str:
    # The setup script pre-populates the project-local Hugging Face cache.  In
    # normal operation the app never needs the network.
    return "1" if (DEFAULT_MODELS_DIR / "hf_cache").is_dir() else "0"


# These must be set before huggingface_hub / open_clip are imported anywhere.
os.environ.setdefault("HF_HOME", str(DEFAULT_MODELS_DIR / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", _offline_default())
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings.

    Defaults are tuned for the verified WSL2 + RTX 3050 + 960x540 camera
    configuration, but every value can be overridden with an environment
    variable (see README).
    """

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    # Camera
    camera_index: int = 0
    # In WSL the V4L2 node numbering can change after a USB re-attach and stale
    # nodes may remain.  Auto mode probes capture nodes and picks a live one.
    camera_auto: bool = True
    camera_width: int = 960
    camera_height: int = 540
    camera_fps: int = 30
    mirror: bool = True

    # Startup behaviour
    auto_start_camera: bool = True
    device: str = "auto"  # auto | cuda | cpu

    # Recognition cadence
    analysis_width: int = 640
    max_analysis_fps: float = 8.0
    stream_fps: int = 20
    jpeg_quality: int = 82

    # Face / human detection
    face_min_confidence: float = 0.70
    face_min_area: float = 0.012  # fraction of frame
    face_max_area: float = 0.90

    # Fish localization / verifier (ImageNet ConvNeXt-Tiny)
    yolo_fish_confidence: float = 0.10
    yolo_min_box_area: float = 0.003
    fish_gate_p: float = 0.18
    fish_gate_top1: float = 0.08
    fish_gate_species_p: float = 0.28

    # Species decision (BioCLIP-2 zero-shot)
    species_min_p: float = 0.45
    species_min_margin: float = 0.08
    species_tie_min_p: float = 0.30
    imagenet_tie_min_confidence: float = 0.25

    # When both a face and a fish are visible, a clearly dominant fish wins.
    # Otherwise a human face takes precedence.
    fish_dominance_min_area: float = 0.10
    fish_dominance_ratio: float = 1.25
    fish_dominance_no_box_p: float = 0.55

    # Temporal smoothing
    smooth_window_seconds: float = 1.5
    smooth_min_votes: int = 2
    keep_previous_seconds: float = 1.0
    stale_seconds: float = 2.5

    # Paths
    models_dir: Path = DEFAULT_MODELS_DIR

    @classmethod
    def from_env(cls) -> "Settings":
        base = cls()
        models_dir = Path(_env_str("MODELS_DIR", str(base.models_dir))).expanduser().resolve()
        # An explicit CAMERA_INDEX is a pin: do not let auto-detection override
        # it unless CAMERA_AUTO is also set explicitly.
        explicit_index = bool(os.environ.get("CAMERA_INDEX"))
        settings = replace(
            base,
            host=_env_str("APP_HOST", base.host),
            port=_env_int("APP_PORT", base.port),
            camera_index=_env_int("CAMERA_INDEX", base.camera_index),
            camera_auto=_env_bool("CAMERA_AUTO", base.camera_auto and not explicit_index),
            camera_width=_env_int("CAMERA_WIDTH", base.camera_width),
            camera_height=_env_int("CAMERA_HEIGHT", base.camera_height),
            camera_fps=_env_int("CAMERA_FPS", base.camera_fps),
            mirror=_env_bool("CAMERA_MIRROR", base.mirror),
            auto_start_camera=_env_bool("AUTO_START_CAMERA", base.auto_start_camera),
            device=_env_str("DEVICE", base.device).lower(),
            analysis_width=_env_int("ANALYSIS_WIDTH", base.analysis_width),
            max_analysis_fps=_env_float("MAX_ANALYSIS_FPS", base.max_analysis_fps),
            stream_fps=_env_int("STREAM_FPS", base.stream_fps),
            jpeg_quality=_env_int("JPEG_QUALITY", base.jpeg_quality),
            face_min_confidence=_env_float("FACE_MIN_CONFIDENCE", base.face_min_confidence),
            face_min_area=_env_float("FACE_MIN_AREA", base.face_min_area),
            yolo_fish_confidence=_env_float("YOLO_FISH_CONFIDENCE", base.yolo_fish_confidence),
            fish_gate_p=_env_float("FISH_GATE_P", base.fish_gate_p),
            fish_gate_top1=_env_float("FISH_GATE_TOP1", base.fish_gate_top1),
            fish_gate_species_p=_env_float(
                "FISH_GATE_SPECIES_P", base.fish_gate_species_p
            ),
            species_min_p=_env_float("SPECIES_MIN_P", base.species_min_p),
            species_min_margin=_env_float("SPECIES_MIN_MARGIN", base.species_min_margin),
            species_tie_min_p=_env_float("SPECIES_TIE_MIN_P", base.species_tie_min_p),
            imagenet_tie_min_confidence=_env_float(
                "IMAGENET_TIE_MIN_CONFIDENCE", base.imagenet_tie_min_confidence
            ),
            fish_dominance_min_area=_env_float("FISH_DOMINANCE_MIN_AREA", base.fish_dominance_min_area),
            fish_dominance_ratio=_env_float("FISH_DOMINANCE_RATIO", base.fish_dominance_ratio),
            fish_dominance_no_box_p=_env_float("FISH_DOMINANCE_NO_BOX_P", base.fish_dominance_no_box_p),
            smooth_window_seconds=_env_float("SMOOTH_WINDOW_SECONDS", base.smooth_window_seconds),
            smooth_min_votes=_env_int("SMOOTH_MIN_VOTES", base.smooth_min_votes),
            keep_previous_seconds=_env_float("KEEP_PREVIOUS_SECONDS", base.keep_previous_seconds),
            stale_seconds=_env_float("STALE_SECONDS", base.stale_seconds),
            models_dir=models_dir,
        )
        # Keep the HF cache inside the project by default, even when MODELS_DIR
        # is overridden.  This call is intentionally made before models are
        # imported by the rest of the application.
        os.environ.setdefault("HF_HOME", str(models_dir / "hf_cache"))
        return settings

    @property
    def face_model_path(self) -> Path:
        return self.models_dir / "face_detection_yunet_2023mar.onnx"

    @property
    def fish_model_path(self) -> Path:
        return self.models_dir / "fish_detector.pt"

    @property
    def imagenet_model_path(self) -> Path:
        return self.models_dir / "convnext_tiny_imagenet.pth"

    def validate(self) -> None:
        if self.port <= 0 or self.port > 65535:
            raise ValueError("APP_PORT must be between 1 and 65535")
        if self.camera_index < 0:
            raise ValueError("CAMERA_INDEX must be >= 0")
        if self.camera_width < 160 or self.camera_height < 120:
            raise ValueError("Camera resolution is too small")
        if not 0.0 <= self.face_min_confidence <= 1.0:
            raise ValueError("FACE_MIN_CONFIDENCE must be in [0, 1]")
        if not 0.0 <= self.fish_gate_p <= 1.0:
            raise ValueError("FISH_GATE_P must be in [0, 1]")
        if not 0.0 <= self.fish_gate_species_p <= 1.0:
            raise ValueError("FISH_GATE_SPECIES_P must be in [0, 1]")
        if not 0.0 <= self.species_min_p <= 1.0:
            raise ValueError("SPECIES_MIN_P must be in [0, 1]")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("DEVICE must be auto, cuda or cpu")
        if self.smooth_min_votes < 1:
            raise ValueError("SMOOTH_MIN_VOTES must be >= 1")

    def as_public_dict(self) -> dict:
        return {
            "camera_index": self.camera_index,
            "camera_auto": self.camera_auto,
            "camera_resolution": [self.camera_width, self.camera_height],
            "mirror": self.mirror,
            "device": self.device,
            "thresholds": {
                "face_min_confidence": self.face_min_confidence,
                "fish_gate_p": self.fish_gate_p,
                "fish_gate_top1": self.fish_gate_top1,
                "fish_gate_species_p": self.fish_gate_species_p,
                "species_min_p": self.species_min_p,
                "species_min_margin": self.species_min_margin,
                "yolo_fish_confidence": self.yolo_fish_confidence,
            },
            "smoothing": {
                "window_seconds": self.smooth_window_seconds,
                "min_votes": self.smooth_min_votes,
            },
        }
