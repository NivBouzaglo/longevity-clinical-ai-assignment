"""Thin async SQLite access helper.

The database is tiny, but the endpoints are ``async`` — so use ``aiosqlite``,
NOT the stdlib ``sqlite3`` (whose blocking calls would stall the event loop).

This module is intentionally minimal: it hands you a connection with a row
factory. Writing the actual SELECT/INSERT statements against the schema in
``data/DATA_DICTIONARY.md`` is your job (see ``app/services/risk.py`` and
``app/api/v1/endpoints.py``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiosqlite

from ..core.config import settings


@asynccontextmanager
async def open_db() -> AsyncIterator[aiosqlite.Connection]:
    """Yield an aiosqlite connection with ``Row`` access (dict-like rows).

    Usage:
        async with open_db() as db:
            async with db.execute("SELECT ... WHERE patient_id = ?", (pid,)) as cur:
                row = await cur.fetchone()
    """
    conn = await aiosqlite.connect(settings.patient_db_path)
    conn.row_factory = aiosqlite.Row
    try:
        yield conn
    finally:
        await conn.close()
