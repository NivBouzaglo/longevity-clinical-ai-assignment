"""Shared pytest fixtures for the backend tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from backend.app.main import app

# backend/tests/conftest.py -> repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATE_DB_SCRIPT = REPO_ROOT / "data" / "generate_db.py"


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    """An async HTTP client that talks to the app in-process (no server needed)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def reset_db():
    """Regenerate ``data/patient_db.db`` to its pristine seeded state around a test.

    ``get_current_risks`` appends real rows to a real SQLite file (not an
    in-memory test DB), so any test that calls it mutates on-disk state that
    other tests (and manual runs) implicitly assume is pristine. Request this
    fixture in any test that calls ``get_current_risks`` to make it independent
    of execution order and to leave the file exactly as it started, even if the
    test fails partway through.
    """

    def _regenerate() -> None:
        subprocess.run(
            [sys.executable, str(GENERATE_DB_SCRIPT)],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
        )

    _regenerate()
    yield
    _regenerate()
