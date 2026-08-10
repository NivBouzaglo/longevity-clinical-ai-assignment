"""Acceptance tests for the clinical API.

Run:  uv run pytest
Note: most of the risk tests hit the live MLflow model server (see GUIDE.md),
since get_current_risks calls it — start it first. The 502 test mocks
``httpx.AsyncClient.post`` instead, since that error path can't reasonably be
triggered against a real server mid test-suite-run. Tests that call
get_current_risks depend on the ``reset_db`` fixture (see conftest.py) to keep
``data/patient_db.db`` pristine across runs.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from unittest.mock import AsyncMock, patch

import httpx
from httpx import AsyncClient

from backend.app.core.config import settings

RISK_CODES = {"CVD", "T2DM", "CKD", "CLD", "DEMENTIA"}
BANDS = {"low", "borderline", "intermediate", "high"}


async def test_health(client: AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_biomarkers_known_patient(client: AsyncClient) -> None:
    r = await client.get("/api/v1/get_current_biomarkers", params={"patient_id": "P001"})
    assert r.status_code == 200
    body = r.json()
    assert body["patient_id"] == "P001"
    # Maya Cohen's eGFR in the mock data is 102.
    assert body["biomarkers"]["egfr_ml_min_1_73m2"] == 102


async def test_unknown_patient_returns_404(client: AsyncClient) -> None:
    r = await client.get("/api/v1/get_current_biomarkers", params={"patient_id": "NOPE"})
    assert r.status_code == 404


async def test_biomarkers_second_known_patient(client: AsyncClient) -> None:
    """A different patient (P002, David Levi) should return his own data, not P001's

    (guards against hardcoded/copy-pasted values).
    """
    r = await client.get("/api/v1/get_current_biomarkers", params={"patient_id": "P002"})
    assert r.status_code == 200
    body = r.json()
    assert body["patient_id"] == "P002"
    assert body["name"] == "David Levi"
    assert body["age_years"] == 68
    assert body["sex"] == "male"
    bio = body["biomarkers"]
    assert bio["patient_id"] == "P002"
    assert bio["measured_at"] == "2026-06-15"
    assert bio["systolic_bp"] == 158
    assert bio["diastolic_bp"] == 94
    assert bio["hdl_cholesterol_mgdl"] == 36
    assert bio["egfr_ml_min_1_73m2"] == 74
    assert bio["urine_dipstick_protein"] == "trace"


async def test_biomarkers_response_shape(client: AsyncClient) -> None:
    """Top-level demographic fields and the nested biomarker snapshot are typed

    and populated for a known patient (P004, Avraham Friedman).
    """
    r = await client.get("/api/v1/get_current_biomarkers", params={"patient_id": "P004"})
    assert r.status_code == 200
    body = r.json()

    assert body["patient_id"] == "P004"
    assert body["name"] == "Avraham Friedman"
    assert isinstance(body["age_years"], int)
    assert body["age_years"] == 72
    assert body["sex"] == "male"

    bio = body["biomarkers"]
    assert bio["measured_at"] == "2026-06-15"
    numeric_fields = [
        "systolic_bp",
        "diastolic_bp",
        "total_cholesterol_mgdl",
        "hdl_cholesterol_mgdl",
        "ldl_cholesterol_mgdl",
        "triglycerides_mgdl",
        "hba1c_percent",
        "fasting_glucose_mgdl",
        "egfr_ml_min_1_73m2",
        "creatinine_mgdl",
        "uacr_mg_g",
        "ggt_u_l",
        "alt_u_l",
        "ast_u_l",
    ]
    for field in numeric_fields:
        assert isinstance(bio[field], (int, float)), f"{field} should be numeric"
    assert isinstance(bio["urine_dipstick_protein"], str)
    # P004 is the designed CKD patient: low eGFR with proteinuria on the dipstick.
    assert bio["egfr_ml_min_1_73m2"] == 52
    assert bio["urine_dipstick_protein"] == "1+"


async def test_biomarkers_missing_patient_id_returns_422(client: AsyncClient) -> None:
    """Omitting the required patient_id query param is a validation error, not a 404/500."""
    r = await client.get("/api/v1/get_current_biomarkers")
    assert r.status_code == 422


async def test_biomarkers_empty_patient_id_returns_404(client: AsyncClient) -> None:
    """An empty-but-present patient_id doesn't match any patient and is a 404, not a crash."""
    r = await client.get("/api/v1/get_current_biomarkers", params={"patient_id": ""})
    assert r.status_code == 404


async def test_risks_returns_five_bands(client: AsyncClient, reset_db) -> None:
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


async def test_risks_are_appended(client: AsyncClient, reset_db) -> None:
    """Calling get_current_risks should persist today's values to the risks log."""

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


async def test_risks_response_shape_metadata(client: AsyncClient, reset_db) -> None:
    """Every RiskResult has populated model metadata and a sane ISO computed_at.

    Per data/DATA_DICTIONARY.md: model_name/model_version are populated for every
    risk, and time_horizon_years is an int for every risk except T2DM (documented
    as "NULL for T2DM screening").
    """
    r = await client.get("/api/v1/get_current_risks", params={"patient_id": "P001"})
    assert r.status_code == 200
    body = r.json()

    # Top-level computed_at is a real, timezone-aware ISO timestamp.
    computed_at = datetime.fromisoformat(body["computed_at"])
    assert computed_at.tzinfo is not None

    assert {x["risk_code"] for x in body["risks"]} == RISK_CODES
    for x in body["risks"]:
        assert isinstance(x["model_name"], str) and x["model_name"]
        assert isinstance(x["model_version"], str) and x["model_version"]
        # Each risk's own computed_at must also parse as ISO.
        datetime.fromisoformat(x["computed_at"])
        if x["risk_code"] == "T2DM":
            assert x["time_horizon_years"] is None
        else:
            assert isinstance(x["time_horizon_years"], int)
            assert x["time_horizon_years"] > 0


async def test_risks_trend_excludes_current_and_matches_seeded_history(
    client: AsyncClient, reset_db
) -> None:
    """trends[risk_code] surfaces the seeded prior points, never the just-computed value.

    Regression test for a dedup bug where the reused ("deduped") current row used
    to leak into its own trend as a duplicate of the current result.
    """
    con = sqlite3.connect(settings.patient_db_path)
    con.row_factory = sqlite3.Row
    try:
        seeded = con.execute(
            "SELECT computed_at, probability, risk_band FROM risks "
            "WHERE patient_id='P003' AND risk_code='T2DM' ORDER BY computed_at ASC, id ASC"
        ).fetchall()
    finally:
        con.close()
    assert len(seeded) == 2  # pristine seed: two back-dated rows per (patient, risk)

    r1 = await client.get("/api/v1/get_current_risks", params={"patient_id": "P003"})
    assert r1.status_code == 200
    body1 = r1.json()
    trend1 = body1["trends"]["T2DM"]
    current1 = next(x for x in body1["risks"] if x["risk_code"] == "T2DM")

    # Trend is exactly the two seeded points, oldest -> newest, and none of them
    # is the just-computed current value.
    assert [(p["computed_at"], p["probability"], p["risk_band"]) for p in trend1] == [
        (row["computed_at"], row["probability"], row["risk_band"]) for row in seeded
    ]
    assert all(p["computed_at"] != current1["computed_at"] for p in trend1)

    # Calling again immediately (same static biomarkers -> dedup path) must not
    # change the trend: the just-deduped "current" row must not leak in as a
    # duplicate history point.
    r2 = await client.get("/api/v1/get_current_risks", params={"patient_id": "P003"})
    assert r2.status_code == 200
    trend2 = r2.json()["trends"]["T2DM"]
    assert trend2 == trend1


async def test_risks_repeat_call_does_not_duplicate_rows(client: AsyncClient, reset_db) -> None:
    """Two immediate calls for a patient whose biomarkers haven't changed insert

    each risk row once, not twice (dedup keyed on the computed inputs payload).
    """

    def row_count(patient_id: str) -> int:
        con = sqlite3.connect(settings.patient_db_path)
        try:
            return con.execute(
                "SELECT COUNT(*) FROM risks WHERE patient_id = ?", (patient_id,)
            ).fetchone()[0]
        finally:
            con.close()

    before = row_count("P005")
    r1 = await client.get("/api/v1/get_current_risks", params={"patient_id": "P005"})
    assert r1.status_code == 200
    after_first = row_count("P005")
    assert after_first == before + 5  # one new row per risk_code

    r2 = await client.get("/api/v1/get_current_risks", params={"patient_id": "P005"})
    assert r2.status_code == 200
    after_second = row_count("P005")
    assert after_second == after_first  # dedup: no growth on the immediate repeat call


async def test_risks_unknown_patient_returns_404(client: AsyncClient) -> None:
    r = await client.get("/api/v1/get_current_risks", params={"patient_id": "NOPE"})
    assert r.status_code == 404


async def test_risks_missing_patient_id_returns_422(client: AsyncClient) -> None:
    """Omitting the required patient_id query param is a validation error, not a 404/500."""
    r = await client.get("/api/v1/get_current_risks")
    assert r.status_code == 422


async def test_risks_returns_502_when_mlflow_unreachable(client: AsyncClient) -> None:
    """If the MLflow model server is unreachable, the endpoint surfaces a 502, not

    a 500 crash or a hang. Mocked (rather than hit against the real server) since
    there's no reasonable way to stop/start a live MLflow server mid test-suite-run.
    """
    with patch(
        "httpx.AsyncClient.post",
        new=AsyncMock(side_effect=httpx.ConnectError("connection refused")),
    ):
        r = await client.get("/api/v1/get_current_risks", params={"patient_id": "P001"})
    assert r.status_code == 502
