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

from ..schemas import BiomarkersResponse, RisksResponse


async def get_current_biomarkers(patient_id: str) -> BiomarkersResponse:
    """Return the latest biomarker snapshot for a patient (404 if unknown)."""
    raise NotImplementedError("Implement get_current_biomarkers (backend exercise).")


async def get_current_risks(patient_id: str) -> RisksResponse:
    """Compute the five risks live, append them to the risks log, and return them."""
    raise NotImplementedError("Implement get_current_risks (backend exercise).")
