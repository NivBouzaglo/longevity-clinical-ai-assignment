"""API v1 routes.

Two endpoints, both wired to ``app.services.risk`` (which is where the real work
lives). Kept thin: parse input, call the service, return a typed response.

The MCP server calls these over HTTP, so keep the contract stable and the errors
meaningful (404 for an unknown patient, 502 if the model server is unreachable).
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from ...schemas import BiomarkersResponse, RisksResponse
from ...services import risk as risk_service

router = APIRouter(prefix="/api/v1", tags=["clinical"])


@router.get("/get_current_biomarkers", response_model=BiomarkersResponse)
async def get_current_biomarkers(
    patient_id: str = Query(..., description="Patient identifier, e.g. P001"),
) -> BiomarkersResponse:
    """Return the latest biomarker snapshot for a patient."""
    return await risk_service.get_current_biomarkers(patient_id)


@router.get("/get_current_risks", response_model=RisksResponse)
async def get_current_risks(
    patient_id: str = Query(..., description="Patient identifier, e.g. P001"),
) -> RisksResponse:
    """Compute the five clinical risks in real time, persist them, and return them."""
    return await risk_service.get_current_risks(patient_id)
