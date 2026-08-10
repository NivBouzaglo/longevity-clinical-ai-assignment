"""Shared pytest fixtures for the backend tests."""

from __future__ import annotations

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from backend.app.main import app


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    """An async HTTP client that talks to the app in-process (no server needed)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
