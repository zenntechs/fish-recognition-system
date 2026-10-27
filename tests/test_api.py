"""HTTP contract tests using an injected fake service (no GPU/camera)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.service import StreamFrame

DISPLAY = {
    "line1": "Fish: True",
    "line2": 'Type of Fish: "Clownfish"',
}


class FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def _state(self) -> dict[str, Any]:
        return {
            "service": {
                "phase": "ready",
                "ready": True,
                "models_ready": True,
                "error": None,
                "uptime": 12.0,
                "started_at": time.time() - 12,
            },
            "status_text": "Live recognition",
            "camera": {
                "status": "running",
                "running": True,
                "index": 0,
                "width": 960,
                "height": 540,
                "fps": 30.0,
                "device": "/dev/video0",
                "message": "Live",
            },
            "result": {
                "fish": True,
                "type_of_fish": "Clownfish",
                "display": DISPLAY,
                "confidence": 0.91,
                "subject": "fish",
                "stability": 0.86,
                "reason": "bioclip_confident",
                "updated_at": time.time(),
                "face": None,
                "fish_box": {"x1": 0.3, "y1": 0.3, "x2": 0.7, "y2": 0.7},
                "p_fish": 0.95,
                "species_top3": [{"name": "Clownfish", "probability": 0.91}],
                "stale": False,
            },
            "stats": {
                "camera_fps": 30.0,
                "analysis_fps": 7.5,
                "stream_fps": 20.0,
                "inference_ms": 80.0,
                "frames_analyzed": 10,
                "analysis_errors": 0,
                "last_analysis_at": time.time(),
                "frame_age": 0.02,
            },
            "models": {
                "device": "cuda",
                "device_name": "Test GPU",
                "warning": None,
                "species_count": 74,
            },
            "settings": {"camera_index": 0},
            "server_time": time.time(),
        }

    def state(self) -> dict[str, Any]:
        return self._state()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "service": self._state()["service"],
            "status_text": "Live recognition",
            "camera": {**self._state()["camera"], "devices": ["/dev/video0"]},
            "models": {"ready": True},
            "model_files": {},
            "environment": {"python": "3.14.0"},
            "settings": {},
            "species_count": 74,
        }

    def species(self) -> dict[str, Any]:
        return {
            "count": 2,
            "species": [
                {"name": "Clownfish", "scientific": "Amphiprion ocellaris"},
                {"name": "Goldfish", "scientific": "Carassius auratus"},
            ],
        }

    def start_camera(self, index: int | None = None) -> dict[str, Any]:
        self.calls.append(("start", index))
        return self._state()["camera"]

    def stop_camera(self) -> dict[str, Any]:
        self.calls.append(("stop", None))
        return {**self._state()["camera"], "running": False, "status": "stopped"}

    def retry_camera(self, index: int | None = None) -> dict[str, Any]:
        self.calls.append(("retry", index))
        return self._state()["camera"]

    def snapshot_jpeg(self) -> bytes:
        return b"\xff\xd8fake-jpeg\xff\xd9"

    def latest_jpeg(self) -> StreamFrame:
        return StreamFrame(b"\xff\xd8fake-stream\xff\xd9", 1, time.time())


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    app = create_app(
        Settings(models_dir=tmp_path),
        service=FakeService(),  # type: ignore[arg-type]
        start_service=False,
    )
    with TestClient(app) as test_client:
        yield test_client


def test_dashboard_contains_required_result_markup(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "Fish: —" in body
    assert 'Type of Fish: "—"' in body
    assert "Start Camera" in body
    assert "Stop" in body
    assert "Retry" in body


def test_state_api_lines_are_consistent(client: TestClient) -> None:
    response = client.get("/api/state")
    assert response.status_code == 200
    assert "no-store" in response.headers["cache-control"]
    data = response.json()
    assert data["result"]["fish"] is True
    assert data["result"]["display"] == DISPLAY
    assert data["result"]["display"]["line1"] == "Fish: True"
    assert data["result"]["type_of_fish"] == "Clownfish"
    assert data["stats"]["analysis_fps"] == 7.5


def test_health_and_diagnostics(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    diagnostics = client.get("/api/diagnostics")
    assert diagnostics.status_code == 200
    assert diagnostics.json()["species_count"] == 74


def test_species_api(client: TestClient) -> None:
    response = client.get("/api/species")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["species"][0]["name"] == "Clownfish"


def test_camera_controls(client: TestClient) -> None:
    start = client.post("/api/camera/start", json={"index": 0})
    assert start.status_code == 200
    assert start.json()["ok"] is True
    assert start.json()["state"]["result"]["display"] == DISPLAY

    stop = client.post("/api/camera/stop")
    assert stop.status_code == 200
    assert stop.json()["camera"]["running"] is False

    retry = client.post("/api/camera/retry", json={})
    assert retry.status_code == 200

    service = client.app.state.service
    assert service.calls == [("start", 0), ("stop", None), ("retry", None)]


def test_snapshot_returns_jpeg(client: TestClient) -> None:
    response = client.get("/api/snapshot")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")


def test_stream_returns_mjpeg(client: TestClient) -> None:
    with client.stream("GET", "/api/stream") as response:
        assert response.status_code == 200
        assert "multipart/x-mixed-replace" in response.headers["content-type"]
        chunk = next(response.iter_bytes())
        assert b"--frameboundary" in chunk
        assert b"Content-Type: image/jpeg" in chunk
        assert b"\xff\xd8fake-stream" in chunk


def test_static_assets_exist(client: TestClient) -> None:
    css = client.get("/static/styles.css")
    js = client.get("/static/app.js")
    icon = client.get("/favicon.svg")
    assert css.status_code == 200
    assert js.status_code == 200
    assert icon.status_code == 200
    assert "result-line" in css.text
    assert "/api/state" in js.text
