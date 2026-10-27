"""Threaded, self-healing webcam capture for Linux/WSL2 (V4L2)."""

from __future__ import annotations

import glob
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Settings

LOGGER = logging.getLogger(__name__)


class CameraStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass(frozen=True)
class CameraInfo:
    status: str
    running: bool
    index: int
    width: int
    height: int
    fps: float
    device: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "running": self.running,
            "index": self.index,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 1),
            "device": self.device,
            "message": self.message,
        }


def discover_video_devices() -> list[str]:
    return sorted(glob.glob("/dev/video*"))


def _interface_number(node: Path) -> int:
    """Return the UVC interface number for a video node (1.0 -> 0)."""
    try:
        name = node.resolve().name  # e.g. "1-2:1.0"
        return int(name.rsplit(".", 1)[1])
    except (OSError, ValueError, IndexError):
        return 99


def capture_node_indices() -> list[int]:
    """List capture nodes, colour interfaces first, then newest device.

    ``index`` (sysfs) is 0 for the video capture node and 1 for the metadata
    node of the same device.  UVC cameras often expose the colour interface
    before an IR interface, which is why interface number is the primary sort
    key.

    A dead USB/IP attachment can leave ``/dev/video*`` nodes behind that block
    forever on the first ioctl.  Probing such a node is not safe (a killed
    process can remain stuck in the kernel), so selection relies purely on
    sysfs.  A re-attached camera gets a higher USB device number than the
    orphaned node it replaces, so the newest enumeration wins.
    """
    nodes: list[tuple[int, int, int]] = []
    for path in glob.glob("/dev/video*"):
        match = re.fullmatch(r"/dev/video(\d+)", path)
        if not match:
            continue
        index = int(match.group(1))
        sysfs = Path(f"/sys/class/video4linux/video{index}")
        try:
            if (sysfs / "index").read_text().strip() != "0":
                continue
            interface = _interface_number(sysfs / "device")
            devnum = _usb_devnum(sysfs / "device")
        except OSError:
            continue
        nodes.append((interface, -devnum, index))
    nodes.sort()
    return [index for _, _, index in nodes]


def _usb_devnum(node: Path) -> int:
    """USB device number of the parent device; newer attachments are higher."""
    try:
        return int((node.resolve().parent / "devnum").read_text().strip())
    except (OSError, ValueError):
        return 0


def choose_camera_index() -> int | None:
    """Return the best capture node without touching the device."""
    nodes = capture_node_indices()
    return nodes[0] if nodes else None



def frame_quality(frame: np.ndarray | None) -> str | None:
    """Return a short corruption label, or None for a usable frame.

    The WSL USB/IP path can emit a uniform green "frame" at some resolutions
    (1280x720 in the verified environment).  Treating that as a valid image
    would silently hide a broken camera, so it is rejected explicitly.
    """
    if frame is None or frame.size == 0:
        return "empty"
    small = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    if mean < 6.0:
        return "dark"
    pixels = small.reshape(-1, 3).astype(np.float32)
    b, g, r = (float(v) for v in pixels.mean(axis=0))
    green = (
        (pixels[:, 1] > 1.35 * pixels[:, 2] + 8.0)
        & (pixels[:, 1] > 1.35 * pixels[:, 0] + 8.0)
        & (pixels[:, 1] > 50.0)
    )
    green_fraction = float(green.mean())
    green_std = float(pixels[:, 1].std())
    if green_fraction > 0.90 and g > 60.0 and green_std < 30.0:
        return "corrupt-green"
    return None


def _troubleshooting_message(index: int) -> str:
    devices = discover_video_devices()
    if not devices:
        return (
            "No Linux video device was found. In WSL, the Windows camera must be "
            "attached first.\n\n"
            "1. Open PowerShell as Administrator on Windows.\n"
            "2. Run: usbipd list\n"
            "3. Find the Integrated Camera BUSID, then run:\n"
            "   usbipd attach --wsl --busid <BUSID>\n"
            "4. Return here and press Retry Camera.\n\n"
            "The camera is then unavailable to Windows applications until you "
            "run: usbipd detach --busid <BUSID>"
        )
    return (
        f"Camera index {index} could not be opened. Detected devices: "
        f"{', '.join(devices)}.\n\n"
        "The device may be busy in another application, the selected index may "
        "be an IR/metadata node, or the USB/IP link may have dropped. Try index "
        "0, close other camera apps, or re-attach the camera from PowerShell:\n"
        "   usbipd detach --busid <BUSID>\n"
        "   usbipd attach --wsl --busid <BUSID>"
    )


class CameraManager:
    """Owns one V4L2 capture thread and publishes the latest BGR frame."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()
        self._start_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._status = CameraStatus.STOPPED
        self._message = "Camera is stopped"
        self._index = settings.camera_index
        self._auto = settings.camera_auto
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._frame_time = 0.0
        self._frame_times: deque[float] = deque(maxlen=30)

    # ------------------------------------------------------------------ public
    def start(self, index: int | None = None) -> CameraInfo:
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return self.info()
            if index is not None:
                if index < 0:
                    raise ValueError("Camera index must be >= 0")
                self._index = int(index)
                # An explicit choice (CLI/API) must be honoured, not overridden
                # by auto-detection.
                self._auto = False
            self._stop_event.clear()
            with self._lock:
                self._status = CameraStatus.STARTING
                self._message = f"Opening /dev/video{self._index} ..."
                self._frame = None
                self._fps = 0.0
            self._thread = threading.Thread(
                target=self._run,
                name="camera-capture",
                daemon=True,
            )
            self._thread.start()
        return self.info()

    def stop(self, timeout: float = 5.0) -> CameraInfo:
        self._stop_event.set()
        with self._start_lock:
            thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._lock:
            self._thread = None
            self._status = CameraStatus.STOPPED
            self._message = "Camera is stopped"
            self._frame = None
            self._fps = 0.0
        return self.info()

    def retry(self, index: int | None = None) -> CameraInfo:
        self.stop()
        time.sleep(0.15)
        return self.start(index)

    def read_latest(self) -> tuple[np.ndarray, int, float] | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy(), self._frame_id, self._frame_time

    def info(self) -> CameraInfo:
        with self._lock:
            return CameraInfo(
                status=self._status.value,
                running=self._status is CameraStatus.RUNNING,
                index=self._index,
                width=self._width,
                height=self._height,
                fps=self._fps,
                device=f"/dev/video{self._index}",
                message=self._message,
            )

    @property
    def running(self) -> bool:
        with self._lock:
            return self._status is CameraStatus.RUNNING

    @property
    def status(self) -> CameraStatus:
        with self._lock:
            return self._status

    def last_frame_age(self) -> float:
        with self._lock:
            if self._frame_time <= 0:
                return float("inf")
            return max(0.0, time.time() - self._frame_time)

    # ----------------------------------------------------------------- internal
    def _candidate_configs(self) -> list[tuple[int, int, str]]:
        configs = [
            (self.settings.camera_width, self.settings.camera_height, "MJPG"),
            (960, 540, "MJPG"),
            (640, 480, "MJPG"),
            (1920, 1080, "MJPG"),
            (1280, 720, "MJPG"),
            (640, 480, "YUYV"),
        ]
        seen: set[tuple[int, int, str]] = set()
        result: list[tuple[int, int, str]] = []
        for cfg in configs:
            if cfg not in seen:
                seen.add(cfg)
                result.append(cfg)
        return result

    def _set_status(self, status: CameraStatus, message: str) -> None:
        with self._lock:
            self._status = status
            self._message = message
        if status in {CameraStatus.ERROR, CameraStatus.RECONNECTING}:
            LOGGER.warning("Camera %s: %s", status.value, message)
        else:
            LOGGER.info("Camera %s: %s", status.value, message)

    def _run(self) -> None:
        ever_running = False
        reconnect_failures = 0
        while not self._stop_event.is_set():
            opened = self._open_camera()
            if opened is None:
                reconnect_failures += 1
                if ever_running and reconnect_failures < 3 and not self._stop_event.is_set():
                    self._set_status(
                        CameraStatus.RECONNECTING,
                        "Camera stream was lost; reconnecting ...",
                    )
                    self._stop_event.wait(1.0)
                    continue
                self._set_status(CameraStatus.ERROR, _troubleshooting_message(self._index))
                return

            cap, width, height, fourcc = opened
            ever_running = True
            reconnect_failures = 0
            self._set_status(
                CameraStatus.RUNNING,
                f"Live from /dev/video{self._index} ({width}x{height}, {fourcc})",
            )
            try:
                healthy = self._capture_loop(cap)
            finally:
                cap.release()

            if self._stop_event.is_set():
                break
            if not healthy:
                self._set_status(
                    CameraStatus.RECONNECTING,
                    "Camera stream stopped unexpectedly; reconnecting ...",
                )
                self._stop_event.wait(0.8)

        with self._lock:
            if self._status is not CameraStatus.ERROR:
                self._status = CameraStatus.STOPPED
                self._message = "Camera is stopped"
            self._frame = None
            self._fps = 0.0

    def _open_camera(self) -> tuple[cv2.VideoCapture, int, int, str] | None:
        index = self._index
        if self._auto:
            resolved = choose_camera_index()
            if resolved is None:
                LOGGER.warning("No responsive V4L2 capture device found")
                return None
            if resolved != index:
                LOGGER.info("Camera auto-selected /dev/video%s", resolved)
            index = resolved
            with self._lock:
                self._index = index
        for width, height, fourcc in self._candidate_configs():
            if self._stop_event.is_set():
                return None
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                # CAP_ANY is a useful fallback on some builds.
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    cap.release()
                    continue
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, self.settings.camera_fps)
                # A short queue keeps the displayed frame close to real time.
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Never let a dead USB/IP link block the capture thread for the
                # V4L2 default of 10 seconds per read.
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
            except cv2.error:
                pass

            last: np.ndarray | None = None
            fail_streak = 0
            for _ in range(16):
                if self._stop_event.is_set():
                    cap.release()
                    return None
                ok, frame = cap.read()
                if ok and frame is not None:
                    last = frame
                    fail_streak = 0
                else:
                    fail_streak += 1
                    if fail_streak >= 4:
                        break
            if last is None or fail_streak >= 4:
                cap.release()
                continue
            quality = frame_quality(last)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(last.shape[1])
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(last.shape[0])
            if quality is not None:
                LOGGER.warning(
                    "Rejected camera mode %sx%s %s: %s", width, height, fourcc, quality
                )
                cap.release()
                continue
            with self._lock:
                self._width = actual_w
                self._height = actual_h
            LOGGER.info("Camera opened: /dev/video%s %sx%s %s", index, actual_w, actual_h, fourcc)
            return cap, actual_w, actual_h, fourcc
        return None

    def _capture_loop(self, cap: cv2.VideoCapture) -> bool:
        fail_streak = 0
        bad_frame_streak = 0
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                fail_streak += 1
                if fail_streak >= 8:
                    return False
                time.sleep(0.02)
                continue
            fail_streak = 0
            quality = frame_quality(frame)
            if quality is not None:
                bad_frame_streak += 1
                if bad_frame_streak >= 45:
                    LOGGER.warning("Too many corrupt frames (%s); reconnecting", quality)
                    return False
                continue
            bad_frame_streak = 0
            if self.settings.mirror:
                frame = cv2.flip(frame, 1)

            now = time.time()
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._frame_time = now
                self._frame_times.append(now)
                if len(self._frame_times) >= 2:
                    span = self._frame_times[-1] - self._frame_times[0]
                    if span > 0:
                        self._fps = (len(self._frame_times) - 1) / span
            # V4L2 read() normally paces at the camera FPS.  The tiny sleep
            # prevents a pathological tight loop if a driver returns instantly.
            time.sleep(0.001)
        return True
