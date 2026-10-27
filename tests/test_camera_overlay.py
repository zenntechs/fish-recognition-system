"""Camera frame-quality guards and overlay rendering tests."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from app.camera import (
    CameraManager,
    _interface_number,
    _usb_devnum,
    choose_camera_index,
    frame_quality,
)
from app.config import Settings
from app.decision import Box, FinalResult, SpeciesCandidate
from app.overlay import OverlayMeta, draw_overlay, encode_jpeg, make_placeholder


def test_normal_frame_is_accepted() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(20, 180, 640, dtype=np.uint8)[None, :]
    frame[:, :, 1] = np.linspace(30, 170, 360, dtype=np.uint8)[:, None]
    frame[:, :, 2] = 90
    assert frame_quality(frame) is None


def test_uniform_green_corruption_is_rejected() -> None:
    frame = np.full((360, 640, 3), (0, 255, 0), dtype=np.uint8)
    assert frame_quality(frame) == "corrupt-green"


def test_black_frame_is_rejected() -> None:
    assert frame_quality(np.zeros((360, 640, 3), dtype=np.uint8)) == "dark"


def test_empty_frame_is_rejected() -> None:
    assert frame_quality(np.zeros((0, 0, 3), dtype=np.uint8)) == "empty"
    assert frame_quality(None) == "empty"


def test_overlay_draws_boxes_without_changing_shape() -> None:
    frame = np.full((360, 640, 3), 72, dtype=np.uint8)
    result = FinalResult.fish_result(
        now=time.time(),
        species="Clownfish",
        confidence=0.93,
        stability=0.9,
        fish_box=Box(0.35, 0.25, 0.7, 0.7, 0.88),
        face=Box(0.05, 0.05, 0.15, 0.15, 0.94),
        p_fish=0.96,
        species_top3=(SpeciesCandidate("Clownfish", 0.93),),
    )
    output = draw_overlay(
        frame,
        result,
        OverlayMeta(
            width=640,
            height=360,
            camera_fps=30.0,
            inference_ms=82.0,
            device="cuda",
            status="running",
        ),
    )
    assert output.shape == frame.shape
    assert output.dtype == frame.dtype
    assert np.count_nonzero(output != frame) > 500


def test_overlay_handles_not_fish_without_boxes() -> None:
    frame = np.full((240, 320, 3), 50, dtype=np.uint8)
    output = draw_overlay(frame, FinalResult.not_fish(now=time.time()))
    assert output.shape == frame.shape


def test_placeholder_is_labeled_and_not_blank() -> None:
    image = make_placeholder(640, 360, "CAMERA OFFLINE", "Press Start Camera")
    assert image.shape == (360, 640, 3)
    assert float(image.std()) > 5.0
    jpeg = encode_jpeg(image, 82)
    assert jpeg.startswith(b"\xff\xd8")
    assert jpeg.endswith(b"\xff\xd9")


def test_overlay_rejects_empty_frame() -> None:
    with pytest.raises(ValueError):
        draw_overlay(np.zeros((0, 0, 3), dtype=np.uint8), FinalResult.not_fish(now=0.0))


def test_corrupt_green_detection_survives_compression_noise() -> None:
    green = np.full((360, 640, 3), (8, 250, 6), dtype=np.uint8)
    noise = np.random.default_rng(7).integers(-4, 5, green.shape, dtype=np.int16)
    noisy = np.clip(green.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    assert frame_quality(noisy) == "corrupt-green"
    assert cv2.imencode(".jpg", noisy)[0]


def test_interface_number_reads_uvc_interface(tmp_path) -> None:
    target = tmp_path / "1-2:1.0"
    target.mkdir()
    link = tmp_path / "device"
    link.symlink_to(target)
    assert _interface_number(link) == 0

    target_ir = tmp_path / "1-2:1.2"
    target_ir.mkdir()
    link_ir = tmp_path / "device-ir"
    link_ir.symlink_to(target_ir)
    assert _interface_number(link_ir) == 2


def test_usb_devnum_reads_parent_device(tmp_path) -> None:
    device = tmp_path / "usb1" / "1-2"
    interface = device / "1-2:1.0"
    interface.mkdir(parents=True)
    (device / "devnum").write_text("4\n")
    link = tmp_path / "device"
    link.symlink_to(interface)
    assert _usb_devnum(link) == 4


def test_choose_camera_index_uses_ranked_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.camera as camera_module

    monkeypatch.setattr(camera_module, "capture_node_indices", lambda: [4, 0])
    assert choose_camera_index() == 4


def test_choose_camera_index_returns_none_without_capture_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.camera as camera_module

    monkeypatch.setattr(camera_module, "capture_node_indices", lambda: [])
    assert choose_camera_index() is None


def test_explicit_index_disables_auto_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = CameraManager(Settings(camera_auto=True))
    # Do not start the capture thread; only validate the selection policy.
    monkeypatch.setattr(manager, "_run", lambda: None)
    manager.start(7)
    try:
        assert manager._auto is False
        assert manager.info().index == 7
    finally:
        manager.stop()


def test_camera_index_env_pins_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAMERA_INDEX", "4")
    monkeypatch.delenv("CAMERA_AUTO", raising=False)
    settings = Settings.from_env()
    assert settings.camera_index == 4
    assert settings.camera_auto is False


def test_camera_auto_env_can_be_forced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAMERA_INDEX", "4")
    monkeypatch.setenv("CAMERA_AUTO", "1")
    assert Settings.from_env().camera_auto is True
