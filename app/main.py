"""FastAPI application and MJPEG streaming endpoints."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .service import AppService

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
BOUNDARY = "frameboundary"


class CameraAction(BaseModel):
    index: int | None = Field(default=None, ge=0, le=63)


def _no_cache(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


def create_app(
    settings: Settings | None = None,
    *,
    service: AppService | None = None,
    start_service: bool = True,
) -> FastAPI:
    """Build the ASGI app.

    ``service``/``start_service`` are injectable so API tests can run without
    loading models or touching camera hardware.
    """
    app_settings = settings or Settings.from_env()
    app_settings.validate()
    app_service = service or AppService(app_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if start_service:
            app_service.start()
        try:
            yield
        finally:
            if start_service:
                app_service.shutdown()

    app = FastAPI(
        title="AquaVision Fish Recognition",
        description="Local, real-time fish / human / not-fish webcam recognition.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.service = app_service
    app.state.settings = app_settings
    app.state.start_service = start_service

    static_dir = STATIC_DIR
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def get_service(request: Request) -> AppService:
        return request.app.state.service

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        path = static_dir / "index.html"
        if not path.is_file():
            raise HTTPException(status_code=500, detail="UI files are missing")
        return FileResponse(path, media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/favicon.svg", include_in_schema=False)
    def favicon() -> FileResponse:
        path = static_dir / "favicon.svg"
        if not path.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(path, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/health")
    def health(request: Request) -> JSONResponse:
        svc = get_service(request)
        state = svc.state()
        ok = state["service"]["ready"] and state["camera"]["running"]
        return _no_cache(
            JSONResponse(
                {
                    "ok": ok,
                    "phase": state["service"]["phase"],
                    "camera": state["camera"]["status"],
                    "models_ready": state["service"]["models_ready"],
                }
            )
        )

    @app.get("/api/state")
    def api_state(request: Request) -> JSONResponse:
        return _no_cache(JSONResponse(get_service(request).state()))

    @app.get("/api/diagnostics")
    def api_diagnostics(request: Request) -> JSONResponse:
        return _no_cache(JSONResponse(get_service(request).diagnostics()))

    @app.get("/api/species")
    def api_species(request: Request) -> JSONResponse:
        return _no_cache(JSONResponse(get_service(request).species()))

    @app.post("/api/camera/start")
    def camera_start(request: Request, action: CameraAction | None = None) -> JSONResponse:
        svc = get_service(request)
        index = action.index if action else None
        try:
            info = svc.start_camera(index)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Camera start failed")
            raise HTTPException(status_code=500, detail=f"Camera start failed: {exc}") from exc
        return _no_cache(JSONResponse({"ok": True, "camera": info, "state": svc.state()}))

    @app.post("/api/camera/stop")
    def camera_stop(request: Request) -> JSONResponse:
        svc = get_service(request)
        try:
            info = svc.stop_camera()
        except Exception as exc:
            LOGGER.exception("Camera stop failed")
            raise HTTPException(status_code=500, detail=f"Camera stop failed: {exc}") from exc
        return _no_cache(JSONResponse({"ok": True, "camera": info, "state": svc.state()}))

    @app.post("/api/camera/retry")
    def camera_retry(request: Request, action: CameraAction | None = None) -> JSONResponse:
        svc = get_service(request)
        index = action.index if action else None
        try:
            info = svc.retry_camera(index)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Camera retry failed")
            raise HTTPException(status_code=500, detail=f"Camera retry failed: {exc}") from exc
        return _no_cache(JSONResponse({"ok": True, "camera": info, "state": svc.state()}))

    @app.get("/api/snapshot")
    def snapshot(request: Request) -> Response:
        jpeg = get_service(request).snapshot_jpeg()
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
            },
        )

    @app.get("/api/stream")
    async def stream(request: Request) -> StreamingResponse:
        svc = get_service(request)

        async def generate() -> AsyncIterator[bytes]:
            last_sequence = -1
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    frame = svc.latest_jpeg()
                    if frame.sequence != last_sequence:
                        last_sequence = frame.sequence
                        header = (
                            f"--{BOUNDARY}\r\n"
                            "Content-Type: image/jpeg\r\n"
                            f"Content-Length: {len(frame.jpeg)}\r\n"
                            "X-Timestamp: "
                            f"{frame.created_at:.3f}\r\n\r\n"
                        ).encode("ascii")
                        yield header + frame.jpeg + b"\r\n"
                    await asyncio.sleep(0.015)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - client disconnect races
                LOGGER.debug("MJPEG stream ended", exc_info=True)

        return StreamingResponse(
            generate(),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


# A module-level app keeps `uvicorn app.main:app` working as well as the CLI.
app = create_app()


def run(host: str | None = None, port: int | None = None, log_level: str = "info") -> None:
    import uvicorn

    settings = app.state.settings
    uvicorn.run(
        app,
        host=host or settings.host,
        port=port or settings.port,
        log_level=log_level,
        access_log=False,
    )
