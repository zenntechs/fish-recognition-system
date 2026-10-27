"""Application service: camera lifecycle, analysis thread, and API state.

One background thread performs inference, one renders the MJPEG frames, and
the HTTP layer only reads immutable snapshots.  Every result the UI sees comes
from the same :class:`FinalResult`, so the two required output lines can never
disagree with the JSON fields.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from importlib import metadata
from typing import Any

import cv2

from .camera import CameraManager, CameraStatus, discover_video_devices
from .config import Settings
from .decision import (
    FinalResult,
    FrameOutcome,
    ResultSmoother,
    classify_frame,
)
from .overlay import OverlayMeta, draw_overlay, encode_jpeg, make_placeholder
from .recognition import ModelLoadError, RecognitionPipeline
from .species import SPECIES

LOGGER = logging.getLogger(__name__)


class ServicePhase(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    LOADING_MODELS = "loading_models"
    READY = "ready"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass(frozen=True)
class StreamFrame:
    jpeg: bytes
    sequence: int
    created_at: float


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class AppService:
    """Owns all long-lived resources for one server process."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.camera = CameraManager(settings)
        self.pipeline = RecognitionPipeline(settings)
        self.smoother = ResultSmoother(settings)

        self.phase = ServicePhase.STOPPED
        self.error: str | None = None
        self.started_at: float | None = None
        self.ready_at: float | None = None

        self._lock = threading.RLock()
        self._camera_action_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._loader_thread: threading.Thread | None = None
        self._analysis_thread: threading.Thread | None = None
        self._render_thread: threading.Thread | None = None
        self._started = False

        self._result: FinalResult = self.smoother.current
        self._last_camera_running = False
        self._last_frame_id = -1
        self._last_analysis_wall = 0.0
        self._last_inference_ms = 0.0
        self._last_analysis_at = 0.0
        self._frames_analyzed = 0
        self._analysis_errors = 0
        self._analysis_times: deque[float] = deque(maxlen=24)
        self._render_times: deque[float] = deque(maxlen=24)

        self._jpeg = encode_jpeg(
            make_placeholder(
                settings.camera_width,
                settings.camera_height,
                "CAMERA OFFLINE",
                "Press Start Camera to begin live recognition",
            ),
            settings.jpeg_quality,
        )
        self._jpeg_sequence = 0
        self._jpeg_created_at = time.time()
        self._placeholder_key = ""

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            self._started = True
            self.started_at = time.time()
            self._stop_event.clear()
            self.phase = ServicePhase.STARTING
            self.error = None
            with self._lock:
                self._result = self.smoother.reset(reason="starting")

            self._render_thread = threading.Thread(
                target=self._render_loop,
                name="frame-render",
                daemon=True,
            )
            self._render_thread.start()

            if self.settings.auto_start_camera:
                try:
                    self.camera.start()
                except Exception as exc:  # pragma: no cover - hardware dependent
                    LOGGER.exception("Automatic camera start failed: %s", exc)
                    self.error = f"Camera could not start: {exc}"

            self.phase = ServicePhase.LOADING_MODELS
            with self._lock:
                self._result = self.smoother.reset(reason="models_loading")
            self._loader_thread = threading.Thread(
                target=self._load_models,
                name="model-loader",
                daemon=True,
            )
            self._loader_thread.start()
            LOGGER.info("Service started (camera auto-start=%s)", self.settings.auto_start_camera)

    def _load_models(self) -> None:
        try:
            self.pipeline.load()
        except ModelLoadError as exc:
            LOGGER.error("Model load failed: %s", exc)
            with self._lock:
                self.phase = ServicePhase.ERROR
                self.error = str(exc)
                if not self.camera.running:
                    self._result = self.smoother.reset(reason="models_unavailable")
            return
        except Exception as exc:  # pragma: no cover - hardware dependent
            LOGGER.exception("Unexpected model load failure")
            with self._lock:
                self.phase = ServicePhase.ERROR
                self.error = f"Model load failed: {exc}"
            return

        if self._stop_event.is_set():
            return
        with self._lock:
            if self.phase is not ServicePhase.STOPPING:
                self.phase = ServicePhase.READY
                self.ready_at = time.time()
                if not self.camera.running:
                    self._result = self.smoother.reset(reason="camera_idle")
        self._analysis_thread = threading.Thread(
            target=self._analysis_loop,
            name="recognition",
            daemon=True,
        )
        self._analysis_thread.start()
        LOGGER.info("Recognition thread started")

    def shutdown(self) -> None:
        with self._start_lock:
            if not self._started:
                return
            with self._lock:
                self.phase = ServicePhase.STOPPING
            self._stop_event.set()

        # Release the camera first: it is the resource that must never leak.
        try:
            self.camera.stop()
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("Camera stop during shutdown failed")

        for thread in (self._analysis_thread, self._render_thread):
            if thread is not None and thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=4.0)

        loader = self._loader_thread
        if loader is not None and loader.is_alive():
            # Model loading can take 30s on a cold cache.  The process is
            # already exiting and the loader is a daemon, so do not block the
            # shutdown or unload models underneath it.
            LOGGER.warning("Shutdown while models are still loading; exiting without unload")
        else:
            self.pipeline.unload()

        with self._lock:
            self.phase = ServicePhase.STOPPED
            self._result = self.smoother.reset(reason="service_stopped")
            self._jpeg_sequence += 1
            self._jpeg_created_at = time.time()
        self._started = False
        LOGGER.info("Service stopped; camera released")

    # ------------------------------------------------------------------ camera API
    def start_camera(self, index: int | None = None) -> dict[str, Any]:
        with self._camera_action_lock:
            if index is not None and int(index) < 0:
                raise ValueError("Camera index must be >= 0")
            with self._lock:
                self._last_frame_id = -1
                self._result = self.smoother.reset(reason="camera_starting")
            info = self.camera.start(index)
            return info.as_dict()

    def stop_camera(self) -> dict[str, Any]:
        with self._camera_action_lock:
            info = self.camera.stop()
            with self._lock:
                self._last_frame_id = -1
                self._result = self.smoother.reset(reason="camera_stopped")
            return info.as_dict()

    def retry_camera(self, index: int | None = None) -> dict[str, Any]:
        with self._camera_action_lock:
            if index is not None and int(index) < 0:
                raise ValueError("Camera index must be >= 0")
            with self._lock:
                self._last_frame_id = -1
                self._result = self.smoother.reset(reason="camera_reconnecting")
            info = self.camera.retry(index)
            return info.as_dict()

    # ------------------------------------------------------------------ threads
    def _analysis_loop(self) -> None:
        min_interval = 1.0 / max(0.5, self.settings.max_analysis_fps)
        while not self._stop_event.is_set():
            if not self.pipeline.ready:
                self._stop_event.wait(0.2)
                continue

            if not self.camera.running:
                if self._last_camera_running:
                    self._last_camera_running = False
                    with self._lock:
                        self._result = self.smoother.reset(reason="camera_unavailable")
                self._stop_event.wait(0.08)
                continue
            self._last_camera_running = True

            data = self.camera.read_latest()
            if data is None:
                self._stop_event.wait(0.03)
                continue

            frame, frame_id, _ = data
            if frame_id == self._last_frame_id:
                self._stop_event.wait(0.01)
                continue

            now = time.time()
            if now - self._last_analysis_wall < min_interval:
                self._stop_event.wait(0.008)
                continue
            self._last_analysis_wall = now
            self._last_frame_id = frame_id

            try:
                analysis = self.pipeline.analyze(frame)
            except Exception as exc:  # pragma: no cover - hardware dependent
                self._analysis_errors += 1
                LOGGER.exception("Frame analysis failed: %s", exc)
                self._stop_event.wait(0.4)
                continue

            outcome: FrameOutcome = classify_frame(analysis.evidence, self.settings)
            completed = time.time()
            with self._lock:
                self._result = self.smoother.update(outcome, now=completed)
                self._last_inference_ms = analysis.inference_ms
                self._last_analysis_at = completed
                self._frames_analyzed += 1
                self._analysis_times.append(completed)

    def _render_loop(self) -> None:
        interval = 1.0 / max(1, self.settings.stream_fps)
        last_placeholder_at = 0.0
        while not self._stop_event.is_set():
            started = time.perf_counter()
            camera_info = self.camera.info()
            data = self.camera.read_latest()
            phase = self.phase

            if data is not None and camera_info.running:
                frame, _, _ = data
                with self._lock:
                    result = self._result
                    inference_ms = self._last_inference_ms
                meta = OverlayMeta(
                    width=camera_info.width,
                    height=camera_info.height,
                    camera_fps=camera_info.fps,
                    inference_ms=inference_ms if phase is ServicePhase.READY else 0.0,
                    device=self.pipeline.device if phase is ServicePhase.READY else "",
                    status="running",
                )
                try:
                    image = draw_overlay(frame, result, meta)
                    jpeg = encode_jpeg(image, self.settings.jpeg_quality)
                except Exception:  # pragma: no cover - defensive
                    LOGGER.exception("Overlay rendering failed; serving raw frame")
                    jpeg = encode_jpeg(frame, self.settings.jpeg_quality)
            else:
                key, message, detail = self._placeholder_copy(camera_info, phase)
                # Regenerating the placeholder once a second is enough; it
                # carries a live clock only in the camera HUD, not here.
                if key != self._placeholder_key or time.time() - last_placeholder_at > 1.0:
                    self._placeholder_key = key
                    last_placeholder_at = time.time()
                    jpeg = encode_jpeg(
                        make_placeholder(
                            self.settings.camera_width,
                            self.settings.camera_height,
                            message,
                            detail,
                        ),
                        self.settings.jpeg_quality,
                    )
                else:
                    with self._lock:
                        jpeg = self._jpeg

            now = time.time()
            with self._lock:
                self._jpeg = jpeg
                self._jpeg_sequence += 1
                self._jpeg_created_at = now
                self._render_times.append(now)

            elapsed = time.perf_counter() - started
            self._stop_event.wait(max(0.003, interval - elapsed))

    def _placeholder_copy(self, camera_info: Any, phase: ServicePhase) -> tuple[str, str, str]:
        if phase in {ServicePhase.STARTING, ServicePhase.LOADING_MODELS}:
            return (
                "loading",
                "STARTING RECOGNITION",
                "Loading local models. The live feed appears automatically.",
            )
        if phase is ServicePhase.ERROR:
            return (
                "error",
                "SETUP REQUIRED",
                "Recognition models are unavailable. Open Diagnostics for details.",
            )
        if camera_info.status == CameraStatus.RECONNECTING.value:
            return (
                "reconnecting",
                "RECONNECTING",
                "The USB camera stream was interrupted. Reconnecting automatically.",
            )
        if camera_info.status == CameraStatus.ERROR.value:
            return (
                "camera-error",
                "CAMERA UNAVAILABLE",
                "Open Camera Help for WSL USB setup, then press Retry Camera.",
            )
        if camera_info.status == CameraStatus.STARTING.value:
            return ("connecting", "CONNECTING", "Opening the camera and waiting for frames.")
        return (
            "stopped",
            "CAMERA OFFLINE",
            "Press Start Camera to begin live recognition",
        )

    # ------------------------------------------------------------------ snapshots
    def _effective_result(self, now: float) -> FinalResult:
        """Never present a stale decision as a current live observation."""
        with self._lock:
            result = self._result
        camera_info = self.camera.info()
        if not camera_info.running:
            return replace(
                FinalResult.not_fish(
                    now,
                    reason="camera_not_running",
                    confidence=0.0,
                    stability=0.0,
                ),
                stale=True,
            )
        if self.phase is not ServicePhase.READY:
            return replace(
                FinalResult.not_fish(
                    now,
                    reason="models_unavailable",
                    confidence=0.0,
                    stability=0.0,
                ),
                stale=True,
            )
        age = max(0.0, now - result.updated_at)
        if age > self.settings.stale_seconds:
            return replace(
                FinalResult.not_fish(
                    now,
                    reason="stale_observation",
                    confidence=0.0,
                    stability=0.0,
                ),
                stale=True,
            )
        return result

    @staticmethod
    def _rate(times: deque[float]) -> float:
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        if span <= 0:
            return 0.0
        return (len(times) - 1) / span

    def state(self) -> dict[str, Any]:
        now = time.time()
        camera_info = self.camera.info()
        result = self._effective_result(now)
        with self._lock:
            phase = self.phase
            error = self.error
            frames = self._frames_analyzed
            errors = self._analysis_errors
            inference_ms = self._last_inference_ms
            last_analysis_at = self._last_analysis_at
            analysis_fps = self._rate(self._analysis_times)
            stream_fps = self._rate(self._render_times)
            uptime = now - self.started_at if self.started_at else 0.0
            camera_fps = camera_info.fps

        if phase is ServicePhase.ERROR:
            status_text = "Recognition unavailable"
        elif phase in {ServicePhase.STARTING, ServicePhase.LOADING_MODELS}:
            status_text = "Loading local models"
        elif not camera_info.running:
            status_text = "Camera stopped"
        elif result.stale:
            status_text = "Waiting for a fresh frame"
        else:
            status_text = "Live recognition"

        payload = {
            "service": {
                "phase": phase.value,
                "ready": phase is ServicePhase.READY,
                "models_ready": self.pipeline.ready,
                "error": error,
                "uptime": round(uptime, 1),
                "started_at": self.started_at,
            },
            "status_text": status_text,
            "camera": camera_info.as_dict(),
            "result": result.as_dict(),
            "stats": {
                "camera_fps": round(camera_fps, 1),
                "analysis_fps": round(analysis_fps, 2),
                "stream_fps": round(stream_fps, 1),
                "inference_ms": round(inference_ms, 1),
                "frames_analyzed": frames,
                "analysis_errors": errors,
                "last_analysis_at": last_analysis_at,
                "frame_age": round(self.camera.last_frame_age(), 2)
                if camera_info.running
                else None,
            },
            "models": {
                "device": self.pipeline.device,
                "device_name": self.pipeline.device_name,
                "warning": self.pipeline.warning,
                "species_count": len(SPECIES),
            },
            "settings": self.settings.as_public_dict(),
            "server_time": now,
        }
        return payload

    def diagnostics(self) -> dict[str, Any]:
        devices = discover_video_devices()
        snapshot = self.state()
        model_files = {
            "face_detector": self.settings.face_model_path,
            "fish_detector": self.settings.fish_model_path,
            "imagenet_verifier": self.settings.imagenet_model_path,
            "bioclip_cache": self.settings.models_dir
            / "hf_cache"
            / "hub"
            / "models--imageomics--bioclip-2",
        }
        return {
            "service": snapshot["service"],
            "status_text": snapshot["status_text"],
            "camera": {
                **self.camera.info().as_dict(),
                "devices": devices,
                "config": {
                    "index": self.settings.camera_index,
                    "requested_resolution": [
                        self.settings.camera_width,
                        self.settings.camera_height,
                    ],
                    "mirror": self.settings.mirror,
                    "stream_fps": self.settings.stream_fps,
                },
            },
            "models": self.pipeline.model_info(),
            "model_files": {
                name: {"path": str(path), "present": path.exists()}
                for name, path in model_files.items()
            },
            "environment": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "torch": _package_version("torch"),
                "torchvision": _package_version("torchvision"),
                "ultralytics": _package_version("ultralytics"),
                "open_clip": _package_version("open-clip-torch"),
                "fastapi": _package_version("fastapi"),
                "offline_hub": os.environ.get("HF_HUB_OFFLINE"),
            },
            "settings": self.settings.as_public_dict(),
            "species_count": len(SPECIES),
        }

    def species(self) -> dict[str, Any]:
        return {
            "count": len(SPECIES),
            "species": [
                {"name": item.name, "scientific": item.scientific} for item in SPECIES
            ],
        }

    def latest_jpeg(self) -> StreamFrame:
        with self._lock:
            return StreamFrame(self._jpeg, self._jpeg_sequence, self._jpeg_created_at)

    def snapshot_jpeg(self) -> bytes:
        return self.latest_jpeg().jpeg
