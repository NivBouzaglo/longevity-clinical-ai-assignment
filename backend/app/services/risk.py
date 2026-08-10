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

from datetime import date

from fastapi import HTTPException

from ..db.sqlite import open_db
from ..schemas import BiomarkerSnapshot, BiomarkersResponse, RisksResponse

# Clinic "today" is fixed for deterministic age derivation (see data/DATA_DICTIONARY.md).
CLINIC_TODAY = date(2026, 7, 9)


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


async def get_current_risks(patient_id: str) -> RisksResponse:
    """Compute the five risks live, append them to the risks log, and return them."""
    raise NotImplementedError("Implement get_current_risks (backend exercise).")
