#!/usr/bin/env python3
"""End-to-end live verification against the real camera and models.

This starts the same :class:`AppService` used by the web dashboard, waits for
the models to load and for real frames to be analysed, writes one annotated
snapshot, then shuts the service down and confirms the camera was released.

Usage::

    .venv/bin/python scripts/verify_live.py
    .venv/bin/python scripts/verify_live.py --timeout 180 --output /tmp/live.jpg

Exit code 0 means the live pipeline was observed working.  It never reports a
synthetic or cached prediction as success.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings  # noqa: E402
from app.service import AppService, ServicePhase  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the live AquaVision pipeline")
    parser.add_argument("--timeout", type=float, default=150.0, help="seconds to wait for live frames")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "live-verification.jpg",
        help="where to write the annotated snapshot",
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    service = AppService(settings)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    last_note: tuple[str, str] | None = None
    success = False
    snapshot_bytes: bytes | None = None
    state: dict | None = None

    print(f"Starting live pipeline (device={settings.device}, timeout={args.timeout:.0f}s)")
    started = time.time()
    try:
        service.start()
        deadline = started + args.timeout
        while time.time() < deadline:
            state = service.state()
            phase = state["service"]["phase"]
            camera = state["camera"]
            stats = state["stats"]
            result = state["result"]

            note = f"phase={phase} camera={camera['status']}"
            if note != last_note:
                print(f"  {note}  frames={stats['frames_analyzed']}")
                last_note = note

            if phase == "error":
                print(f"ERROR: {state['service']['error']}", file=sys.stderr)
                return 1

            if (
                phase == "ready"
                and camera["running"]
                and stats["frames_analyzed"] >= 3
                and stats["frame_age"] is not None
                and stats["frame_age"] < 1.5
                and not result["stale"]
            ):
                snapshot_bytes = service.snapshot_jpeg()
                success = True
                break
            time.sleep(0.5)
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        print("\nInterrupted", file=sys.stderr)
        return 130
    finally:
        service.shutdown()

    released = not service.camera.running and service.phase is ServicePhase.STOPPED

    if state is None:
        print("ERROR: the service produced no state", file=sys.stderr)
        return 1

    print("\nLive verification")
    print("-----------------")
    print(f"  models          : {state['models']['device']} · {state['models']['device_name']}")
    print(f"  species set     : {state['models']['species_count']}")
    print(f"  camera          : {state['camera']['device']} "
          f"({state['camera']['width']}x{state['camera']['height']}, {state['camera']['fps']:.1f} fps)")
    print(f"  display line 1  : {state['result']['display']['line1']}")
    print(f"  display line 2  : {state['result']['display']['line2']}")
    print(f"  confidence      : {state['result']['confidence']:.1%}")
    print(f"  analysis rate   : {state['stats']['analysis_fps']:.1f} fps")
    print(f"  inference       : {state['stats']['inference_ms']:.0f} ms")
    print(f"  frames analyzed : {state['stats']['frames_analyzed']}")
    print(f"  camera released : {released}")

    if not success or snapshot_bytes is None:
        print(f"\nFAILED: no live, non-stale frame within {args.timeout:.0f}s", file=sys.stderr)
        return 1

    output.write_bytes(snapshot_bytes)
    if not (snapshot_bytes.startswith(b"\xff\xd8") and snapshot_bytes.endswith(b"\xff\xd9")):
        print("FAILED: snapshot is not a valid JPEG", file=sys.stderr)
        return 1
    print(f"  snapshot        : {output} ({len(snapshot_bytes) / 1024:.0f} KiB)")
    print(f"\nPASS in {time.time() - started:.1f}s")
    return 0 if released else 1


if __name__ == "__main__":
    raise SystemExit(main())
