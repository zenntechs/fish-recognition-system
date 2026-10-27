"""Command line entry point: ``python -m app``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="Local real-time fish recognition from a webcam.",
    )
    parser.add_argument("--host", default=None, help="Bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="HTTP port (default 8000)")
    parser.add_argument("--camera", type=int, default=None, help="V4L2 camera index (default 0)")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default=None,
        help="Inference device (default auto)",
    )
    parser.add_argument("--width", type=int, default=None, help="Requested camera width")
    parser.add_argument("--height", type=int, default=None, help="Requested camera height")
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Do not open the camera automatically on startup",
    )
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error"),
        default="info",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.host:
        os.environ["APP_HOST"] = args.host
    if args.port:
        os.environ["APP_PORT"] = str(args.port)
    if args.camera is not None:
        os.environ["CAMERA_INDEX"] = str(args.camera)
        os.environ["CAMERA_AUTO"] = "0"
    if args.device:
        os.environ["DEVICE"] = args.device
    if args.width:
        os.environ["CAMERA_WIDTH"] = str(args.width)
    if args.height:
        os.environ["CAMERA_HEIGHT"] = str(args.height)
    if args.no_auto_start:
        os.environ["AUTO_START_CAMERA"] = "0"

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("ultralytics", "open_clip", "huggingface_hub", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from .config import Settings
    from .main import create_app
    from .service import AppService

    try:
        settings = Settings.from_env()
        settings.validate()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    service = AppService(settings)
    application = create_app(settings, service=service)

    print()
    print("  AQUAVISION  |  local fish recognition")
    print(f"  Dashboard:  http://{settings.host}:{settings.port}")
    camera_label = (
        "auto-detect" if settings.camera_auto else f"/dev/video{settings.camera_index}"
    )
    print(f"  Camera:     {camera_label} ({settings.camera_width}x{settings.camera_height})")
    print(f"  Device:     {settings.device}")
    print(f"  Models:     {settings.models_dir}")
    print("  Stop:       Ctrl+C (releases the camera)")
    print()

    import uvicorn

    try:
        uvicorn.run(
            application,
            host=settings.host,
            port=settings.port,
            log_level=args.log_level,
            access_log=False,
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
