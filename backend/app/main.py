"""FastAPI application entrypoint.

Run (from the repo root):
    uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8001 --reload

``/health`` works out of the box so you can confirm the app boots; the two
clinical endpoints under ``/api/v1`` return 501 until you implement them.
"""

from __future__ import annotations

from fastapi import FastAPI

from .api.v1.endpoints import router as v1_router
from .core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.include_router(v1_router)

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name}

    return app


app = create_app()
