"""
Generate the five risk models shipped with the assignment (models/*.pkl).

Each model is a plain scikit-learn ``LogisticRegression`` — a simple linear model
with a ``predict_proba`` method and discoverable ``feature_names_in_``. We do NOT
fit them on data; instead we set directionally-correct coefficients by hand
(e.g. CVD risk rises with age, systolic BP, smoking; falls with HDL) and then
auto-calibrate each model's scale + intercept so that a "healthy" reference
patient maps to a low probability and a "high-risk" reference patient maps to a
high probability. Everything in between is monotonic and bounded in (0, 1).

These are deliberately SURROGATES of the published instruments (AHA PREVENT, the
ADA diabetes risk test, the Framingham CKD score, CLivD, CAIDE) — good enough to
demonstrate real-time, biomarker-driven risk inference, not clinically validated.

The candidate must discover each model's expected inputs (``model.feature_names_in_``)
and build the payload themselves — that mapping is intentionally NOT provided here.

Run:  uv run python models/generate_models.py
"""

from __future__ import annotations

import pickle
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODELS_DIR = Path(__file__).resolve().parent
MODEL_VERSION = "1.0.0"


def _logit(p: float) -> float:
    return log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def calibrate(feature_names, base_coef, healthy, high_risk, p_low, p_high):
    """Solve for a scale + intercept so the two reference patients hit p_low/p_high.

    Returns (calibrated_coef_dict, intercept). Keeps the hand-set signs/ratios in
    ``base_coef`` intact (multiplies them all by a single positive scale).
    """
    l_healthy = sum(base_coef[f] * healthy[f] for f in feature_names)
    l_high = sum(base_coef[f] * high_risk[f] for f in feature_names)
    if l_high <= l_healthy:
        raise ValueError("high-risk anchor must have a larger raw logit than healthy anchor")
    scale = (_logit(p_high) - _logit(p_low)) / (l_high - l_healthy)
    intercept = _logit(p_low) - scale * l_healthy
    coef = {f: scale * base_coef[f] for f in feature_names}
    return coef, intercept


def build_model(feature_names, coef, intercept, metadata) -> LogisticRegression:
    """Assemble a ready-to-predict LogisticRegression from explicit coefficients."""
    clf = LogisticRegression()
    clf.classes_ = np.array([0, 1])
    clf.coef_ = np.array([[coef[f] for f in feature_names]], dtype=float)
    clf.intercept_ = np.array([intercept], dtype=float)
    clf.n_features_in_ = len(feature_names)
    clf.feature_names_in_ = np.asarray(feature_names, dtype=object)
    # Metadata travels inside the pickle (discoverable via ``clf.metadata_``).
    clf.metadata_ = dict(metadata, model_version=MODEL_VERSION)
    return clf


# --- Model specifications ---------------------------------------------------
# For each model: the feature vector the model expects (order matters), the
# hand-set coefficient signs/magnitudes in RAW units, two reference patients,
# and the target probability band used for calibration.
MODELS = [
    {
        "filename": "prevent_cvd.pkl",
        "metadata": {
            "risk_code": "CVD", "model_name": "prevent_cvd",
            "outcome": "10-year total cardiovascular disease risk",
            "time_horizon_years": 10,
            "surrogate_of": "AHA PREVENT (2023)",
        },
        "features": ["age_years", "sex_male", "total_cholesterol_mgdl", "hdl_cholesterol_mgdl",
                     "systolic_bp", "bp_treated", "on_statin", "diabetes", "current_smoker",
                     "bmi", "egfr"],
        "base_coef": {"age_years": 0.06, "sex_male": 0.5, "total_cholesterol_mgdl": 0.008,
                      "hdl_cholesterol_mgdl": -0.02, "systolic_bp": 0.02, "bp_treated": 0.3,
                      "on_statin": -0.2, "diabetes": 0.6, "current_smoker": 0.7, "bmi": 0.02,
                      "egfr": -0.01},
        "healthy": {"age_years": 35, "sex_male": 0, "total_cholesterol_mgdl": 180,
                    "hdl_cholesterol_mgdl": 65, "systolic_bp": 112, "bp_treated": 0,
                    "on_statin": 0, "diabetes": 0, "current_smoker": 0, "bmi": 22, "egfr": 100},
        "high_risk": {"age_years": 68, "sex_male": 1, "total_cholesterol_mgdl": 260,
                      "hdl_cholesterol_mgdl": 35, "systolic_bp": 158, "bp_treated": 1,
                      "on_statin": 1, "diabetes": 0, "current_smoker": 1, "bmi": 30, "egfr": 72},
        "p_low": 0.03, "p_high": 0.45,
    },
    {
        "filename": "ada_t2dm.pkl",
        "metadata": {
            "risk_code": "T2DM", "model_name": "ada_t2dm",
            "outcome": "elevated risk of type 2 diabetes (screening)",
            "time_horizon_years": None,
            "surrogate_of": "ADA Type 2 Diabetes Risk Test",
        },
        "features": ["age_years", "sex_male", "bmi", "family_history_diabetes",
                     "hypertension", "physically_active", "gestational_diabetes"],
        "base_coef": {"age_years": 0.03, "sex_male": 0.1, "bmi": 0.12,
                      "family_history_diabetes": 0.7, "hypertension": 0.5,
                      "physically_active": -0.5, "gestational_diabetes": 0.6},
        "healthy": {"age_years": 30, "sex_male": 0, "bmi": 22, "family_history_diabetes": 0,
                    "hypertension": 0, "physically_active": 1, "gestational_diabetes": 0},
        "high_risk": {"age_years": 57, "sex_male": 0, "bmi": 33, "family_history_diabetes": 1,
                      "hypertension": 1, "physically_active": 0, "gestational_diabetes": 1},
        "p_low": 0.03, "p_high": 0.55,
    },
    {
        "filename": "framingham_ckd.pkl",
        "metadata": {
            "risk_code": "CKD", "model_name": "framingham_ckd",
            "outcome": "10-year risk of incident chronic kidney disease (stage 3+)",
            "time_horizon_years": 10,
            "surrogate_of": "Framingham CKD risk score",
        },
        "features": ["age_years", "diabetes", "hypertension", "proteinuria_trace_plus", "egfr"],
        "base_coef": {"age_years": 0.05, "diabetes": 0.7, "hypertension": 0.5,
                      "proteinuria_trace_plus": 0.9, "egfr": -0.03},
        "healthy": {"age_years": 35, "diabetes": 0, "hypertension": 0,
                    "proteinuria_trace_plus": 0, "egfr": 100},
        "high_risk": {"age_years": 72, "diabetes": 1, "hypertension": 1,
                      "proteinuria_trace_plus": 1, "egfr": 52},
        "p_low": 0.02, "p_high": 0.50,
    },
    {
        "filename": "clivd_cld.pkl",
        "metadata": {
            "risk_code": "CLD", "model_name": "clivd_cld",
            "outcome": "15-year risk of incident chronic liver disease",
            "time_horizon_years": 15,
            "surrogate_of": "CLivD score (with GGT)",
        },
        "features": ["age_years", "sex_male", "alcohol_drinks_per_week", "waist_hip_ratio",
                     "diabetes", "current_smoker", "ggt_u_l"],
        "base_coef": {"age_years": 0.02, "sex_male": 0.4, "alcohol_drinks_per_week": 0.05,
                      "waist_hip_ratio": 3.0, "diabetes": 0.4, "current_smoker": 0.5,
                      "ggt_u_l": 0.01},
        "healthy": {"age_years": 35, "sex_male": 0, "alcohol_drinks_per_week": 2,
                    "waist_hip_ratio": 0.75, "diabetes": 0, "current_smoker": 0, "ggt_u_l": 18},
        "high_risk": {"age_years": 61, "sex_male": 1, "alcohol_drinks_per_week": 28,
                      "waist_hip_ratio": 1.12, "diabetes": 1, "current_smoker": 1, "ggt_u_l": 145},
        "p_low": 0.02, "p_high": 0.40,
    },
    {
        "filename": "caide_dementia.pkl",
        "metadata": {
            "risk_code": "DEMENTIA", "model_name": "caide_dementia",
            "outcome": "20-year risk of dementia",
            "time_horizon_years": 20,
            "surrogate_of": "CAIDE dementia risk score",
        },
        "features": ["age_years", "sex_male", "education_years", "systolic_bp", "bmi",
                     "total_cholesterol_mgdl", "physically_active"],
        "base_coef": {"age_years": 0.06, "sex_male": 0.2, "education_years": -0.15,
                      "systolic_bp": 0.015, "bmi": 0.05, "total_cholesterol_mgdl": 0.006,
                      "physically_active": -0.4},
        "healthy": {"age_years": 35, "sex_male": 0, "education_years": 18, "systolic_bp": 112,
                    "bmi": 22, "total_cholesterol_mgdl": 180, "physically_active": 1},
        "high_risk": {"age_years": 66, "sex_male": 0, "education_years": 8, "systolic_bp": 156,
                      "bmi": 32, "total_cholesterol_mgdl": 262, "physically_active": 0},
        "p_low": 0.03, "p_high": 0.45,
    },
]


def main() -> None:
    print(f"{'model':16} {'features':>8}  {'p(healthy)':>10}  {'p(high-risk)':>12}")
    for spec in MODELS:
        coef, intercept = calibrate(
            spec["features"], spec["base_coef"], spec["healthy"], spec["high_risk"],
            spec["p_low"], spec["p_high"],
        )
        clf = build_model(spec["features"], coef, intercept, spec["metadata"])
        out = MODELS_DIR / spec["filename"]
        with out.open("wb") as fh:
            pickle.dump(clf, fh)

        # Sanity: re-derive the anchor probabilities from the built model.
        # Pass a DataFrame with named columns (the way callers should) so sklearn
        # validates feature names instead of warning about a bare array.
        def prob(vec):
            X = pd.DataFrame([[vec[f] for f in spec["features"]]], columns=spec["features"])
            return float(clf.predict_proba(X)[0, 1])

        name = spec["metadata"]["model_name"]
        print(f"{name:16} {len(spec['features']):>8}  {prob(spec['healthy']):>10.3f}  "
              f"{prob(spec['high_risk']):>12.3f}   -> {spec['filename']}")


if __name__ == "__main__":
    main()
