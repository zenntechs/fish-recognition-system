"""Local inference pipeline: YuNet faces, YOLO-World fish, ImageNet gate,
and BioCLIP-2 zero-shot species identification."""

from __future__ import annotations

import gc
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import Settings
from .decision import Box, FrameEvidence, SpeciesCandidate, best_fish_box
from .species import (
    IMAGENET_FISH_CLASSES,
    IMAGENET_TO_SPECIES,
    SPECIES,
    SPECIES_BY_NAME,
)

LOGGER = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Analysis:
    evidence: FrameEvidence
    inference_ms: float
    device: str
    frame_width: int
    frame_height: int


@dataclass
class _ImageNetGate:
    p_fish: float
    top1: str
    top1_prob: float
    top5: tuple[tuple[str, float], ...]


class RecognitionPipeline:
    """Thread-confined model pipeline.

    Models are loaded once and reused.  ``analyze`` must not be called
    concurrently; AppService owns a single analysis thread.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()
        self.ready = False
        self.device = "unloaded"
        self.device_name = "unloaded"
        self.load_seconds = 0.0
        self.warning: str | None = None

        self._torch = None
        self._torchvision = None
        self._open_clip = None
        self._yolo = None
        self._yolo_device: int | str = "cpu"
        self._imagenet = None
        self._imagenet_transform = None
        self._imagenet_labels: list[str] = []
        self._fish_indices: list[int] = []
        self._bio = None
        self._bio_preprocess = None
        self._text_features = None
        self._face_detector = None

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        start = time.perf_counter()
        self._check_model_files()
        requested = self.settings.device
        if requested == "auto":
            requested = "cuda" if self._cuda_available() else "cpu"

        if requested == "cuda":
            try:
                self._load_models("cuda")
            except Exception as exc:  # pragma: no cover - hardware dependent
                LOGGER.exception("CUDA model load failed; retrying on CPU: %s", exc)
                self.warning = f"CUDA unavailable ({exc}); using CPU inference."
                self._cleanup_models()
                self._load_models("cpu")
        else:
            self._load_models("cpu")

        self._warmup()
        self.ready = True
        self.load_seconds = time.perf_counter() - start
        LOGGER.info(
            "Models ready on %s (%s) in %.1fs (%d species)",
            self.device,
            self.device_name,
            self.load_seconds,
            len(SPECIES),
        )

    def _cuda_available(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _check_model_files(self) -> None:
        missing = [
            str(path.name)
            for path in (
                self.settings.face_model_path,
                self.settings.fish_model_path,
                self.settings.imagenet_model_path,
            )
            if not path.is_file()
        ]
        hub = self.settings.models_dir / "hf_cache" / "hub" / "models--imageomics--bioclip-2"
        if not hub.is_dir():
            missing.append("models/hf_cache/.../models--imageomics--bioclip-2")
        if missing:
            raise ModelLoadError(
                "Missing model asset(s): "
                + ", ".join(missing)
                + ". Run ./setup.sh (or .venv/bin/python scripts/download_models.py) "
                "to download and cache them."
            )

    def _load_models(self, device: str) -> None:
        import torch
        import torchvision
        import open_clip
        from ultralytics import YOLO

        self._torch = torch
        self._torchvision = torchvision
        self._open_clip = open_clip
        torch.set_grad_enabled(False)
        if device == "cpu":
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
        else:
            torch.backends.cudnn.benchmark = True

        self.device = device
        if device == "cuda":
            self.device_name = torch.cuda.get_device_name(0)
        else:
            self.device_name = platform.processor() or platform.machine() or "CPU"

        # Face detector (OpenCV Zoo YuNet, ONNX, CPU is plenty).
        self._face_detector = cv2.FaceDetectorYN.create(
            str(self.settings.face_model_path),
            "",
            (self.settings.analysis_width, int(self.settings.analysis_width * 9 / 16)),
            0.55,
            0.3,
            5000,
        )

        # Open-vocabulary fish detector.  The saved checkpoint already has the
        # "fish" text embedding baked in, so CLIP is not needed at runtime.
        self._yolo = YOLO(str(self.settings.fish_model_path), task="detect")
        self._yolo_device = 0 if device == "cuda" else "cpu"
        names = getattr(self._yolo, "names", {}) or {}
        if "fish" not in {str(v).lower() for v in names.values()}:
            raise ModelLoadError(
                "The fish detector checkpoint does not contain the 'fish' class. "
                "Re-run scripts/download_models.py."
            )

        # ImageNet verifier.  This is deliberately a general classifier: its
        # aggregate probability over true fish classes is an excellent
        # out-of-distribution gate for webcam scenes.
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        self._imagenet_labels = list(weights.meta["categories"])
        self._fish_indices = [
            i for i, label in enumerate(self._imagenet_labels) if label in IMAGENET_FISH_CLASSES
        ]
        if len(self._fish_indices) < 10:
            raise ModelLoadError("ImageNet fish class list could not be resolved.")
        self._imagenet = torchvision.models.convnext_tiny(weights=None)
        state = torch.load(
            str(self.settings.imagenet_model_path),
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self._imagenet.load_state_dict(state)
        self._imagenet.to(device).eval()
        self._imagenet_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256, antialias=True),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # BioCLIP-2 species model.
        self._bio, _, self._bio_preprocess = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip-2",
            device=device,
        )
        self._bio.eval()
        tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
        prompts: list[str] = []
        for species in SPECIES:
            prompts.extend(species.prompts)
        tokens = tokenizer(prompts).to(device)
        with torch.inference_mode():
            text = self._bio.encode_text(tokens)
            text = text / text.norm(dim=-1, keepdim=True)
            text = text.reshape(len(SPECIES), 4, -1).mean(dim=1)
            text = text / text.norm(dim=-1, keepdim=True)
        self._text_features = text

        if device == "cuda":
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            LOGGER.info("CUDA models loaded (%.2f GiB allocated)", allocated)

    def _warmup(self) -> None:
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        blank[:] = 40
        try:
            self._detect_faces(blank)
            self._classify_imagenet(blank)
            self._classify_species(blank)
            self._detect_fish(blank)
            if self._torch is not None and self.device == "cuda":
                self._torch.cuda.synchronize()
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise ModelLoadError(f"Model warm-up failed: {exc}") from exc

    def _cleanup_models(self) -> None:
        self.ready = False
        self._imagenet = None
        self._bio = None
        self._text_features = None
        self._yolo = None
        self._face_detector = None
        self._bio_preprocess = None
        self._imagenet_transform = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    # ------------------------------------------------------------------- helpers
    @staticmethod
    def _letterbox_square(image: np.ndarray, fill: int = 114) -> np.ndarray:
        h, w = image.shape[:2]
        side = max(h, w)
        canvas = np.full((side, side, 3), fill, dtype=np.uint8)
        y = (side - h) // 2
        x = (side - w) // 2
        canvas[y : y + h, x : x + w] = image
        return canvas

    def _detect_faces(self, frame: np.ndarray) -> tuple[Box, ...]:
        if self._face_detector is None:
            return ()
        h, w = frame.shape[:2]
        self._face_detector.setInputSize((w, h))
        _, faces = self._face_detector.detect(frame)
        if faces is None:
            return ()
        result: list[Box] = []
        for face in faces:
            x, y, fw, fh = (float(v) for v in face[:4])
            confidence = float(face[-1])
            if fw <= 1 or fh <= 1:
                continue
            result.append(
                Box(
                    max(0.0, x / w),
                    max(0.0, y / h),
                    min(1.0, (x + fw) / w),
                    min(1.0, (y + fh) / h),
                    confidence,
                )
            )
        return tuple(result)

    def _detect_fish(self, frame: np.ndarray) -> tuple[Box, ...]:
        if self._yolo is None:
            return ()
        results = self._yolo.predict(
            frame,
            conf=self.settings.yolo_fish_confidence,
            imgsz=640,
            max_det=20,
            verbose=False,
            device=self._yolo_device,
        )
        if not results:
            return ()
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return ()
        h, w = frame.shape[:2]
        result: list[Box] = []
        for box in boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            result.append(
                Box(
                    max(0.0, min(1.0, x1 / w)),
                    max(0.0, min(1.0, y1 / h)),
                    max(0.0, min(1.0, x2 / w)),
                    max(0.0, min(1.0, y2 / h)),
                    conf,
                )
            )
        return tuple(result)

    def _classify_imagenet(self, image: np.ndarray) -> _ImageNetGate:
        assert self._torch is not None
        rgb = cv2.cvtColor(self._letterbox_square(image), cv2.COLOR_BGR2RGB)
        tensor = self._imagenet_transform(Image.fromarray(rgb)).unsqueeze(0)
        tensor = tensor.to(self.device)
        with self._torch.inference_mode():
            probs = self._imagenet(tensor).softmax(dim=-1)[0]
        p_fish = float(probs[self._fish_indices].sum().item())
        values, indices = probs.topk(5)
        top5 = tuple(
            (self._imagenet_labels[int(i)], float(v))
            for v, i in zip(values.tolist(), indices.tolist())
        )
        return _ImageNetGate(p_fish, top5[0][0], top5[0][1], top5)

    def _classify_species(self, image: np.ndarray) -> tuple[SpeciesCandidate, ...]:
        assert self._torch is not None
        rgb = cv2.cvtColor(self._letterbox_square(image), cv2.COLOR_BGR2RGB)
        tensor = self._bio_preprocess(Image.fromarray(rgb)).unsqueeze(0)
        tensor = tensor.to(self.device)
        with self._torch.inference_mode():
            features = self._bio.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            logits = 100.0 * features @ self._text_features.T
            probs = logits.softmax(dim=-1)[0]
        values, indices = probs.topk(5)
        return tuple(
            SpeciesCandidate(SPECIES[int(i)].name, float(v))
            for v, i in zip(values.tolist(), indices.tolist())
        )

    def _crop_for_box(self, frame: np.ndarray, box: Box) -> np.ndarray | None:
        h, w = frame.shape[:2]
        bw = box.width * w
        bh = box.height * h
        pad_x = max(4.0, bw * 0.08)
        pad_y = max(4.0, bh * 0.08)
        x1 = max(0, int(box.x1 * w - pad_x))
        y1 = max(0, int(box.y1 * h - pad_y))
        x2 = min(w, int(box.x2 * w + pad_x))
        y2 = min(h, int(box.y2 * h + pad_y))
        if x2 - x1 < 24 or y2 - y1 < 24:
            return None
        return frame[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------------ inference
    def analyze(self, frame_bgr: np.ndarray) -> Analysis:
        if not self.ready:
            raise RuntimeError("Models are not loaded yet")
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty frame")

        with self._lock:
            started = time.perf_counter()
            h, w = frame_bgr.shape[:2]
            scale = min(1.0, self.settings.analysis_width / float(w))
            if scale < 1.0:
                analysis_frame = cv2.resize(
                    frame_bgr,
                    (self.settings.analysis_width, max(1, int(round(h * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                analysis_frame = frame_bgr

            faces = self._detect_faces(analysis_frame)
            fish_boxes = self._detect_fish(analysis_frame)
            fish_box = best_fish_box(fish_boxes, self.settings)

            full_gate = self._classify_imagenet(analysis_frame)
            crop = self._crop_for_box(analysis_frame, fish_box) if fish_box else None
            crop_gate = self._classify_imagenet(crop) if crop is not None else None

            region = analysis_frame
            gate = full_gate
            if crop is not None and crop_gate is not None:
                # Prefer the detected fish crop when it carries at least as
                # much fish evidence as the whole scene.
                if crop_gate.p_fish >= full_gate.p_fish:
                    region = crop
                    gate = crop_gate

            species: tuple[SpeciesCandidate, ...] = ()
            species_tie: str | None = None
            if gate.p_fish >= self.settings.fish_gate_p:
                species = self._classify_species(region)
                mapped = IMAGENET_TO_SPECIES.get(gate.top1)
                if mapped and gate.top1_prob >= self.settings.imagenet_tie_min_confidence:
                    species_tie = mapped

            evidence = FrameEvidence(
                faces=faces,
                fish_boxes=fish_boxes,
                fish_box=fish_box,
                gate_p_fish=gate.p_fish,
                gate_top1=gate.top1,
                gate_top1_prob=gate.top1_prob,
                species=species,
                species_tie=species_tie,
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            if self._torch is not None and self.device == "cuda":
                # Make the reported latency reflect completed GPU work without
                # adding a full synchronize to every frame.
                pass
            return Analysis(
                evidence=evidence,
                inference_ms=elapsed,
                device=self.device,
                frame_width=analysis_frame.shape[1],
                frame_height=analysis_frame.shape[0],
            )

    def model_info(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "device": self.device,
            "device_name": self.device_name,
            "load_seconds": round(self.load_seconds, 2),
            "warning": self.warning,
            "species_count": len(SPECIES),
            "face_model": "YuNet 2023mar (OpenCV Zoo)",
            "fish_detector": "YOLO-World v2-s (fish prompt)",
            "fish_verifier": "ConvNeXt-Tiny ImageNet-1k",
            "species_model": "BioCLIP-2 ViT-L/14 (zero-shot)",
            "imagenet_fish_classes": sorted(IMAGENET_FISH_CLASSES),
        }

    def unload(self) -> None:
        with self._lock:
            self._cleanup_models()
