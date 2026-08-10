"""Shared pytest fixtures for the mcp-server tests.

These are integration tests: the two real tools (`get_current_biomarkers`,
`get_current_risks`) call out to the live FastAPI backend over HTTP, so there is
no in-process transport to swap in the way ``backend/tests/conftest.py`` does
with ``ASGITransport``. Instead we connect a real ``fastmcp.Client`` to the MCP
server that must already be listening on ``http://127.0.0.1:9000/mcp/`` (see
mcp-server/README.md's "Smoke test" section for the pattern these fixtures
follow).

Run:  uv run pytest mcp-server/tests -v
Requires: backend (:8001), MLflow model server (:5001), and the MCP server
(:9000) all running — see mcp-server/README.md / repo GUIDE.md to start them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

# mcp-server/tests/conftest.py -> repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATE_DB_SCRIPT = REPO_ROOT / "data" / "generate_db.py"

load_dotenv(REPO_ROOT / ".env")

MCP_URL = os.getenv("MCP_TEST_URL", "http://127.0.0.1:9000/mcp/")
MCP_BEARER_TOKEN = os.getenv("MCP_BEARER_TOKEN", "dev-longevity-token-change-me")


@pytest_asyncio.fixture
async def client() -> Client:
    """An authenticated fastmcp.Client connected to the live MCP server."""
    async with Client(MCP_URL, auth=BearerAuth(MCP_BEARER_TOKEN)) as c:
        yield c


@pytest.fixture
def reset_db():
    """Regenerate ``data/patient_db.db`` to its pristine seeded state around a test.

    ``get_current_risks`` (via the backend) appends real rows to the real SQLite
    file, same as the backend's own tests. Request this fixture in any test that
    calls ``get_current_risks`` so the on-disk DB is left exactly as it started,
    even if the test fails partway through, and so tests stay independent of
    execution order (including when run alongside backend/tests in the same
    session, since both suites touch the same file).
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
