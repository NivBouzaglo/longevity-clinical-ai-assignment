"""Acceptance tests for the clinical API.

``test_health`` passes as soon as the scaffold runs. The rest are skipped —
remove the ``@pytest.mark.skip`` decorators as you implement each endpoint and
make them pass. They double as the spec for what "done" means.

Run:  uv run pytest
Note: the risk tests assume the MLflow model server is running (see GUIDE.md),
since get_current_risks calls it. Feel free to mock it instead.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

RISK_CODES = {"CVD", "T2DM", "CKD", "CLD", "DEMENTIA"}
BANDS = {"low", "borderline", "intermediate", "high"}


async def test_health(client: AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.skip(reason="Implement get_current_biomarkers, then remove this skip.")
async def test_biomarkers_known_patient(client: AsyncClient) -> None:
    r = await client.get("/api/v1/get_current_biomarkers", params={"patient_id": "P001"})
    assert r.status_code == 200
    body = r.json()
    assert body["patient_id"] == "P001"
    # Maya Cohen's eGFR in the mock data is 102.
    assert body["biomarkers"]["egfr_ml_min_1_73m2"] == 102


@pytest.mark.skip(reason="Implement the 404 path, then remove this skip.")
async def test_unknown_patient_returns_404(client: AsyncClient) -> None:
    r = await client.get("/api/v1/get_current_biomarkers", params={"patient_id": "NOPE"})
    assert r.status_code == 404


@pytest.mark.skip(reason="Implement get_current_risks, then remove this skip.")
async def test_risks_returns_five_bands(client: AsyncClient) -> None:
    r = await client.get("/api/v1/get_current_risks", params={"patient_id": "P004"})
    assert r.status_code == 200
    risks = r.json()["risks"]
    assert {x["risk_code"] for x in risks} == RISK_CODES
    for x in risks:
        assert 0.0 <= x["probability"] <= 1.0
        assert x["risk_band"] in BANDS
    # P004 (Avraham Friedman) is the designed CKD patient — should read high.
    ckd = next(x for x in risks if x["risk_code"] == "CKD")
    assert ckd["risk_band"] == "high"


@pytest.mark.skip(reason="Implement the risks append, then remove this skip.")
async def test_risks_are_appended(client: AsyncClient) -> None:
    """Calling get_current_risks should persist today's values to the risks log."""
    import sqlite3

    from backend.app.core.config import settings

    def row_count() -> int:
        con = sqlite3.connect(settings.patient_db_path)
        try:
            return con.execute(
                "SELECT COUNT(*) FROM risks WHERE patient_id='P002' AND computed_at >= date('now')"
            ).fetchone()[0]
        finally:
            con.close()

    await client.get("/api/v1/get_current_risks", params={"patient_id": "P002"})
    assert row_count() >= 5
