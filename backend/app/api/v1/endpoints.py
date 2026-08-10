"""API v1 routes.

Two endpoints — both currently return HTTP 501 (Not Implemented). Implement them
by wiring in ``app.services.risk`` (which is where the real work lives). Keep them
thin: parse input, call the service, return a typed response.

The MCP server calls these over HTTP, so keep the contract stable and the errors
meaningful (404 for an unknown patient, 502 if the model server is unreachable).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...schemas import BiomarkersResponse, RisksResponse

router = APIRouter(prefix="/api/v1", tags=["clinical"])


@router.get("/get_current_biomarkers", response_model=BiomarkersResponse)
async def get_current_biomarkers(
    patient_id: str = Query(..., description="Patient identifier, e.g. P001"),
) -> BiomarkersResponse:
    """Return the latest biomarker snapshot for a patient."""
    # TODO: return await risk_service.get_current_biomarkers(patient_id)
    raise HTTPException(
        status_code=501,
        detail="Not implemented — wire this to app.services.risk.get_current_biomarkers",
    )


@router.get("/get_current_risks", response_model=RisksResponse)
async def get_current_risks(
    patient_id: str = Query(..., description="Patient identifier, e.g. P001"),
) -> RisksResponse:
    """Compute the five clinical risks in real time, persist them, and return them."""
    # TODO: return await risk_service.get_current_risks(patient_id)
    raise HTTPException(
        status_code=501,
        detail="Not implemented — wire this to app.services.risk.get_current_risks",
    )
