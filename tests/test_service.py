"""Service snapshot tests without camera hardware or model loading."""

from __future__ import annotations

import time
from pathlib import Path

from app.camera import CameraInfo, CameraStatus
from app.config import Settings
from app.decision import Box, FinalResult, SpeciesCandidate
from app.service import AppService, ServicePhase


class FakeCamera:
    def __init__(self, running: bool = True):
        self.running = running
        self.status = CameraStatus.RUNNING if running else CameraStatus.STOPPED
        self.info_calls = 0

    def info(self) -> CameraInfo:
        self.info_calls += 1
        return CameraInfo(
            status=self.status.value,
            running=self.running,
            index=0,
            width=960 if self.running else 0,
            height=540 if self.running else 0,
            fps=30.0 if self.running else 0.0,
            device="/dev/video0",
            message="Live" if self.running else "Camera is stopped",
        )

    def last_frame_age(self) -> float:
        return 0.02 if self.running else float("inf")


def build_service(tmp_path: Path, running: bool = True) -> tuple[AppService, FakeCamera]:
    service = AppService(Settings(models_dir=tmp_path))
    camera = FakeCamera(running=running)
    service.camera = camera  # type: ignore[assignment]
    service.phase = ServicePhase.READY
    service.pipeline.ready = True
    service.pipeline.device = "cuda"
    service.pipeline.device_name = "Test GPU"
    service.started_at = time.time() - 65
    now = time.time()
    service._result = FinalResult.fish_result(  # noqa: SLF001 - snapshot fixture
        now=now,
        species="Clownfish",
        confidence=0.91,
        stability=0.86,
        fish_box=Box(0.3, 0.25, 0.7, 0.75, 0.88),
        face=None,
        p_fish=0.95,
        species_top3=(
            SpeciesCandidate("Clownfish", 0.91),
            SpeciesCandidate("Goldfish", 0.03),
        ),
    )
    service._analysis_times.extend([now - 0.5, now])  # noqa: SLF001
    return service, camera


def test_state_presents_fish_and_matching_display(tmp_path: Path) -> None:
    service, _ = build_service(tmp_path)
    state = service.state()

    assert state["service"]["phase"] == "ready"
    assert state["camera"]["running"] is True
    assert state["result"]["fish"] is True
    assert state["result"]["type_of_fish"] == "Clownfish"
    assert state["result"]["display"] == {
        "line1": "Fish: True",
        "line2": 'Type of Fish: "Clownfish"',
    }
    assert state["models"]["species_count"] >= 70
    assert state["stats"]["analysis_fps"] > 0
    assert state["status_text"] == "Live recognition"


def test_state_never_shows_a_fish_when_camera_is_stopped(tmp_path: Path) -> None:
    service, camera = build_service(tmp_path, running=False)
    camera.status = CameraStatus.STOPPED
    state = service.state()

    assert state["camera"]["running"] is False
    assert state["result"]["fish"] is False
    assert state["result"]["type_of_fish"] == "Not Fish"
    assert state["result"]["stale"] is True
    assert state["result"]["display"]["line2"] == 'Type of Fish: "Not Fish"'


def test_state_marks_stale_observations(tmp_path: Path) -> None:
    service, _ = build_service(tmp_path)
    service._result = FinalResult.fish_result(  # noqa: SLF001
        now=time.time() - 30,
        species="Goldfish",
        confidence=0.9,
        stability=0.9,
    )
    state = service.state()
    assert state["result"]["fish"] is False
    assert state["result"]["reason"] == "stale_observation"
    assert state["result"]["stale"] is True


def test_state_exposes_model_and_camera_diagnostics(tmp_path: Path) -> None:
    service, _ = build_service(tmp_path)
    diagnostics = service.diagnostics()
    assert diagnostics["environment"]["python"]
    assert diagnostics["camera"]["devices"] is not None
    assert diagnostics["models"]["ready"] is True
    assert "face_detector" in diagnostics["model_files"]
    assert diagnostics["species_count"] >= 70


def test_shutdown_before_start_is_safe(tmp_path: Path) -> None:
    service = AppService(Settings(models_dir=tmp_path))
    service.shutdown()
    assert service.phase is ServicePhase.STOPPED


def test_species_payload_is_curated() -> None:
    service = AppService(Settings())
    payload = service.species()
    assert payload["count"] == len(payload["species"])
    names = {item["name"] for item in payload["species"]}
    assert {"Clownfish", "Goldfish", "Great white shark", "Tuna", "Sole", "Hake"} <= names
    assert "Human" not in names
    assert "Not Fish" not in names
