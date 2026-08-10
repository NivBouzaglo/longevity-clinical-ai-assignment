"""Pydantic response models for the API.

These are a STARTING POINT that defines the shape the MCP tools (and the
assistant) will consume. Extend or adjust them as you implement the endpoints —
but keep responses typed and predictable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BiomarkerSnapshot(BaseModel):
    """Latest labs/vitals for a patient (subset shown; include what you need)."""

    patient_id: str
    measured_at: str
    systolic_bp: float | None = None
    diastolic_bp: float | None = None
    total_cholesterol_mgdl: float | None = None
    hdl_cholesterol_mgdl: float | None = None
    ldl_cholesterol_mgdl: float | None = None
    triglycerides_mgdl: float | None = None
    hba1c_percent: float | None = None
    fasting_glucose_mgdl: float | None = None
    egfr_ml_min_1_73m2: float | None = None
    creatinine_mgdl: float | None = None
    uacr_mg_g: float | None = None
    urine_dipstick_protein: str | None = None
    ggt_u_l: float | None = None
    alt_u_l: float | None = None
    ast_u_l: float | None = None


class BiomarkersResponse(BaseModel):
    patient_id: str
    name: str
    age_years: int
    sex: str
    biomarkers: BiomarkerSnapshot


class RiskResult(BaseModel):
    """A single freshly-computed risk."""

    risk_code: str = Field(examples=["CVD", "T2DM", "CKD", "CLD", "DEMENTIA"])
    probability: float = Field(ge=0.0, le=1.0)
    risk_band: str = Field(examples=["low", "borderline", "intermediate", "high"])
    model_name: str
    model_version: str | None = None
    time_horizon_years: int | None = None
    computed_at: str


class RiskTrendPoint(BaseModel):
    computed_at: str
    probability: float
    risk_band: str


class RisksResponse(BaseModel):
    patient_id: str
    name: str
    computed_at: str
    risks: list[RiskResult]
    # Optional: prior computed values per risk_code so the assistant can show a trend.
    trends: dict[str, list[RiskTrendPoint]] = Field(default_factory=dict)
