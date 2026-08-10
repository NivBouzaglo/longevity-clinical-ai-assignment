"""Risk computation service — THE core logic you implement.

This is the heart of the backend exercise. Nothing here is implemented; the
functions describe the contract and raise ``NotImplementedError``.

What ``get_current_risks`` needs to do, end to end:

  1. Load the patient's demographics + latest biomarkers from SQLite
     (``app.db.sqlite.open_db``). 404 if the patient does not exist.
  2. For each of the five models, build the exact feature payload it expects.
     Discover the features from the model itself — ``model.feature_names_in_`` —
     and derive values (age, BMI, waist-hip ratio, 0/1 flags) from the raw
     columns. Units matter (see data/DATA_DICTIONARY.md and models/README.md).
  3. Call the MLflow model server (``settings.mlflow_url``) to get a probability
     per model. Use an async HTTP client (``httpx.AsyncClient``) and fire the
     calls concurrently (``asyncio.gather``) — they are independent.
  4. Map each probability to a risk band and assemble ``RiskResult`` objects.
  5. APPEND one row per risk to the ``risks`` table (this is what lets the
     assistant show a trend over time). Store ``inputs_json`` for auditability.
     Avoid polluting the trend with duplicates — only insert when the inputs
     changed since the last stored row for that (patient, model). (Bonus points
     for noticing that a GET that writes is an HTTP-semantics smell and handling
     it deliberately.)
  6. Return current risks plus, optionally, the prior points as ``trends``.

You decide how to split this across helpers; the signatures below are a
suggestion, not a requirement.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timezone

import httpx
from fastapi import HTTPException

from ..core.config import settings
from ..db.sqlite import open_db
from ..schemas import (
    BiomarkerSnapshot,
    BiomarkersResponse,
    RiskResult,
    RisksResponse,
    RiskTrendPoint,
)

# Clinic "today" is fixed for deterministic age derivation (see data/DATA_DICTIONARY.md).
CLINIC_TODAY = date(2026, 7, 9)

# How many prior (pre-this-call) history points to surface per risk_code in
# ``RisksResponse.trends``. The seeded history is short (two rows per risk), but
# in general this log grows unbounded, and the assistant only needs enough
# points to describe a trend, not the full audit trail.
TREND_HISTORY_LIMIT = 5

# risk_code -> (mlflow router "model" param / model_name, model_version, time_horizon_years).
# The router "model" param and the stored model_name are the same string for all
# five models here, so one field does double duty.
_MODEL_SPECS: list[tuple[str, str, str, int | None]] = [
    ("CVD", "prevent_cvd", "1.0.0", 10),
    ("T2DM", "ada_t2dm", "1.0.0", None),
    ("CKD", "framingham_ckd", "1.0.0", 10),
    ("CLD", "clivd_cld", "1.0.0", 15),
    ("DEMENTIA", "caide_dementia", "1.0.0", 20),
]

# Exact feature names + order each model expects (models/README.md "Input contracts").
_FEATURE_COLUMNS: dict[str, list[str]] = {
    "CVD": [
        "age_years", "sex_male", "total_cholesterol_mgdl", "hdl_cholesterol_mgdl",
        "systolic_bp", "bp_treated", "on_statin", "diabetes", "current_smoker",
        "bmi", "egfr",
    ],
    "T2DM": [
        "age_years", "sex_male", "bmi", "family_history_diabetes", "hypertension",
        "physically_active", "gestational_diabetes",
    ],
    "CKD": ["age_years", "diabetes", "hypertension", "proteinuria_trace_plus", "egfr"],
    "CLD": [
        "age_years", "sex_male", "alcohol_drinks_per_week", "waist_hip_ratio",
        "diabetes", "current_smoker", "ggt_u_l",
    ],
    "DEMENTIA": [
        "age_years", "sex_male", "education_years", "systolic_bp", "bmi",
        "total_cholesterol_mgdl", "physically_active",
    ],
}

# Banding thresholds (data/DATA_DICTIONARY.md "Risk banding thresholds"), checked
# from the highest band down so each probability falls into exactly one bucket.
_BAND_THRESHOLDS: list[tuple[float, str]] = [
    (0.35, "high"),
    (0.20, "intermediate"),
    (0.10, "borderline"),
]


def _age_years(date_of_birth: str, today: date = CLINIC_TODAY) -> int:
    """Whole years between an ISO ``date_of_birth`` and ``today``."""
    dob = date.fromisoformat(date_of_birth)
    years = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        years -= 1
    return years


async def get_current_biomarkers(patient_id: str) -> BiomarkersResponse:
    """Return the latest biomarker snapshot for a patient (404 if unknown)."""
    async with open_db() as db:
        async with db.execute(
            "SELECT patient_id, first_name, last_name, date_of_birth, sex "
            "FROM demographics WHERE patient_id = ?",
            (patient_id,),
        ) as cur:
            demo = await cur.fetchone()

        if demo is None:
            raise HTTPException(status_code=404, detail=f"Unknown patient_id: {patient_id!r}")

        async with db.execute(
            "SELECT * FROM biomarkers WHERE patient_id = ? "
            "ORDER BY measured_at DESC, id DESC LIMIT 1",
            (patient_id,),
        ) as cur:
            bio = await cur.fetchone()

        if bio is None:
            raise HTTPException(
                status_code=404, detail=f"No biomarkers on file for patient_id: {patient_id!r}"
            )

    return BiomarkersResponse(
        patient_id=demo["patient_id"],
        name=f"{demo['first_name']} {demo['last_name']}",
        age_years=_age_years(demo["date_of_birth"]),
        sex=demo["sex"],
        biomarkers=BiomarkerSnapshot(
            patient_id=bio["patient_id"],
            measured_at=bio["measured_at"],
            systolic_bp=bio["systolic_bp"],
            diastolic_bp=bio["diastolic_bp"],
            total_cholesterol_mgdl=bio["total_cholesterol_mgdl"],
            hdl_cholesterol_mgdl=bio["hdl_cholesterol_mgdl"],
            ldl_cholesterol_mgdl=bio["ldl_cholesterol_mgdl"],
            triglycerides_mgdl=bio["triglycerides_mgdl"],
            hba1c_percent=bio["hba1c_percent"],
            fasting_glucose_mgdl=bio["fasting_glucose_mgdl"],
            egfr_ml_min_1_73m2=bio["egfr_ml_min_1_73m2"],
            creatinine_mgdl=bio["creatinine_mgdl"],
            uacr_mg_g=bio["uacr_mg_g"],
            urine_dipstick_protein=bio["urine_dipstick_protein"],
            ggt_u_l=bio["ggt_u_l"],
            alt_u_l=bio["alt_u_l"],
            ast_u_l=bio["ast_u_l"],
        ),
    )


def _band(probability: float) -> str:
    """Map a probability to its risk band per the DATA_DICTIONARY.md thresholds."""
    for threshold, band in _BAND_THRESHOLDS:
        if probability >= threshold:
            return band
    return "low"


def _build_feature_values(demo, bio) -> dict[str, float]:
    """Derive the full union of model input features from a demographics + biomarkers row.

    Each model only consumes a subset (see ``_FEATURE_COLUMNS``); computing the
    union once avoids re-deriving age/BMI/etc. per model.
    """
    gestational_diabetes = demo["gestational_diabetes"]
    if gestational_diabetes is None:
        # NULL for male patients in the DB (see DATA_DICTIONARY.md) — they never
        # have this risk factor, so the model input is 0, not missing.
        gestational_diabetes = 0

    return {
        "age_years": _age_years(demo["date_of_birth"]),
        "bmi": demo["weight_kg"] / (demo["height_cm"] / 100) ** 2,
        "waist_hip_ratio": demo["waist_cm"] / demo["hip_cm"],
        "sex_male": 1 if demo["sex"] == "male" else 0,
        "current_smoker": 1 if demo["smoking_status"] == "current" else 0,
        "proteinuria_trace_plus": 0 if bio["urine_dipstick_protein"] == "negative" else 1,
        "bp_treated": demo["on_bp_medication"],
        "on_statin": demo["on_statin"],
        "diabetes": demo["hx_diabetes"],
        "hypertension": demo["hx_hypertension"],
        "physically_active": demo["physical_activity_active"],
        "family_history_diabetes": demo["family_history_diabetes"],
        "gestational_diabetes": gestational_diabetes,
        "alcohol_drinks_per_week": demo["alcohol_drinks_per_week"],
        "education_years": demo["education_years"],
        "systolic_bp": bio["systolic_bp"],
        "total_cholesterol_mgdl": bio["total_cholesterol_mgdl"],
        "hdl_cholesterol_mgdl": bio["hdl_cholesterol_mgdl"],
        "ggt_u_l": bio["ggt_u_l"],
        "egfr": bio["egfr_ml_min_1_73m2"],
    }


async def _call_mlflow(client: httpx.AsyncClient, model: str, columns: list[str], values: list[float]) -> float:
    """POST a single-row ``dataframe_split`` payload to the router and return the probability."""
    payload = {
        "dataframe_split": {"columns": columns, "data": [values]},
        "params": {"model": model},
    }
    try:
        resp = await client.post(settings.mlflow_url, json=payload, timeout=30.0)
        resp.raise_for_status()
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502, detail=f"MLflow model server unreachable at {settings.mlflow_url}: {exc}"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"MLflow model server returned {exc.response.status_code}: {exc.response.text}",
        ) from exc

    data = resp.json()
    try:
        return float(data["predictions"][0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502, detail=f"Unexpected MLflow response shape: {data!r}"
        ) from exc


async def get_current_risks(patient_id: str) -> RisksResponse:
    """Compute the five risks live, append them to the risks log, and return them."""
    async with open_db() as db:
        async with db.execute(
            "SELECT patient_id, first_name, last_name, date_of_birth, sex, height_cm, "
            "weight_kg, waist_cm, hip_cm, education_years, smoking_status, "
            "alcohol_drinks_per_week, physical_activity_active, family_history_diabetes, "
            "hx_diabetes, hx_hypertension, gestational_diabetes, on_bp_medication, on_statin "
            "FROM demographics WHERE patient_id = ?",
            (patient_id,),
        ) as cur:
            demo = await cur.fetchone()

        if demo is None:
            raise HTTPException(status_code=404, detail=f"Unknown patient_id: {patient_id!r}")

        async with db.execute(
            "SELECT * FROM biomarkers WHERE patient_id = ? "
            "ORDER BY measured_at DESC, id DESC LIMIT 1",
            (patient_id,),
        ) as cur:
            bio = await cur.fetchone()

        if bio is None:
            raise HTTPException(
                status_code=404, detail=f"No biomarkers on file for patient_id: {patient_id!r}"
            )

        feature_values = _build_feature_values(demo, bio)

        # Build one payload per model up front so the concurrent calls below only
        # have to POST — no more feature-derivation work in the gather.
        specs = []
        for risk_code, model, model_version, horizon in _MODEL_SPECS:
            columns = _FEATURE_COLUMNS[risk_code]
            values = [feature_values[c] for c in columns]
            inputs_json = json.dumps(dict(zip(columns, values)), sort_keys=True)
            specs.append((risk_code, model, model_version, horizon, columns, values, inputs_json))

        # The five models are independent — fire all requests concurrently.
        async with httpx.AsyncClient() as client:
            probabilities = await asyncio.gather(
                *(_call_mlflow(client, spec[1], spec[4], spec[5]) for spec in specs)
            )

        now = datetime.now(timezone.utc).isoformat()
        results: list[RiskResult] = []
        trends: dict[str, list[RiskTrendPoint]] = {}

        for (risk_code, model, model_version, horizon, _columns, _values, inputs_json), probability in zip(
            specs, probabilities
        ):
            band = _band(probability)

            async with db.execute(
                "SELECT probability, risk_band, computed_at, inputs_json FROM risks "
                "WHERE patient_id = ? AND risk_code = ? ORDER BY computed_at DESC, id DESC",
                (patient_id, risk_code),
            ) as cur:
                history = await cur.fetchall()

            latest = history[0] if history else None

            # Dedup: get_current_risks is a GET that writes (append to the trend
            # log), which is an HTTP-semantics smell — the assistant may call it
            # repeatedly (e.g. every chat turn) and demographics/biomarkers rarely
            # change between calls. Re-inserting an identical row each time would
            # spam the trend with near-duplicates that don't reflect any real
            # clinical change. So: only INSERT when the computed feature payload
            # differs from the most recent stored row for this (patient, risk); if
            # it's unchanged, treat the existing row as "current" and touch nothing.
            if latest is not None and latest["inputs_json"] == inputs_json:
                probability = latest["probability"]
                band = latest["risk_band"]
                computed_at = latest["computed_at"]
            else:
                computed_at = now
                await db.execute(
                    "INSERT INTO risks (patient_id, risk_code, probability, risk_band, "
                    "model_name, model_version, time_horizon_years, computed_at, inputs_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        patient_id,
                        risk_code,
                        probability,
                        band,
                        model,
                        model_version,
                        horizon,
                        computed_at,
                        inputs_json,
                    ),
                )

            results.append(
                RiskResult(
                    risk_code=risk_code,
                    probability=probability,
                    risk_band=band,
                    model_name=model,
                    model_version=model_version,
                    time_horizon_years=horizon,
                    computed_at=computed_at,
                )
            )

            # ``history`` holds every prior row (newest first). On a dedup no-op,
            # history[0] is the same row being reused as "current" above, so drop
            # it before slicing — otherwise it leaks into the trend as a duplicate
            # of the current result. Cap and reverse to a chronological
            # (oldest -> newest) trend line, capped at the last TREND_HISTORY_LIMIT
            # points so the response stays small even once the log has accumulated
            # many entries.
            prior_history = (
                history[1:] if (latest is not None and latest["inputs_json"] == inputs_json) else history
            )
            trends[risk_code] = [
                RiskTrendPoint(
                    computed_at=row["computed_at"],
                    probability=row["probability"],
                    risk_band=row["risk_band"],
                )
                for row in reversed(prior_history[:TREND_HISTORY_LIMIT])
            ]

        await db.commit()

    return RisksResponse(
        patient_id=demo["patient_id"],
        name=f"{demo['first_name']} {demo['last_name']}",
        computed_at=now,
        risks=results,
        trends=trends,
    )
