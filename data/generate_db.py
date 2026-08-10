"""
Generate the mock patient SQLite database: data/patient_db.db

This script is the reproducible source of truth for the mock clinic data that
ships with the assignment. It creates three tables — `demographics`,
`biomarkers`, and `risks` — and seeds them with 8 fictional patients whose
values are hand-designed to be clinically consistent and to span the risk space
(one clearly-healthy patient, one high-CVD smoker, a CKD patient, etc.).

IMPORTANT (for assignment authors): this file contains ONLY literal patient data
and the schema. It deliberately contains NO payload-construction or risk-scoring
logic — figuring out how to turn these columns into a model payload is the
candidate's job (see backend/ and models/README.md).

The `risks` table is seeded with a few BACK-DATED rows per patient so a trend
already exists on day one; the candidate's `get_current_risks` endpoint appends
today's freshly-computed row on top of that history. The historical probabilities
below are illustrative values chosen to trend toward each patient's current risk
band — they are plain data, not model outputs.

Run:  uv run python data/generate_db.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "patient_db.db"

# "Today" for this mock clinic. Ages and the seeded history are anchored here so
# the dataset is deterministic regardless of the real wall-clock date.
CLINIC_TODAY = "2026-07-09"
LAST_VISIT = "2026-06-15"  # when the current biomarker snapshot was measured


SCHEMA = """
DROP TABLE IF EXISTS risks;
DROP TABLE IF EXISTS biomarkers;
DROP TABLE IF EXISTS demographics;

-- Patient identity + stable attributes, lifestyle, and medical-history flags.
-- These feed the demographic / lifestyle inputs of the risk models.
CREATE TABLE demographics (
    patient_id              TEXT PRIMARY KEY,
    mrn                     TEXT UNIQUE NOT NULL,        -- medical record number
    first_name              TEXT NOT NULL,
    last_name               TEXT NOT NULL,
    date_of_birth           TEXT NOT NULL,               -- ISO YYYY-MM-DD (derive age vs CLINIC_TODAY)
    sex                     TEXT NOT NULL CHECK (sex IN ('male','female')),
    height_cm               REAL NOT NULL,               -- with weight_kg -> BMI
    weight_kg               REAL NOT NULL,
    waist_cm                REAL,                        -- with hip_cm -> waist-hip ratio (WHR)
    hip_cm                  REAL,
    education_years         INTEGER,                     -- CAIDE dementia input
    smoking_status          TEXT CHECK (smoking_status IN ('never','former','current')),
    alcohol_drinks_per_week REAL,                        -- CLivD input
    physical_activity_active INTEGER CHECK (physical_activity_active IN (0,1)),
    family_history_diabetes INTEGER CHECK (family_history_diabetes IN (0,1)),
    hx_diabetes             INTEGER CHECK (hx_diabetes IN (0,1)),
    hx_hypertension         INTEGER CHECK (hx_hypertension IN (0,1)),
    gestational_diabetes    INTEGER,                     -- 0/1 for female; NULL where not applicable
    on_bp_medication        INTEGER CHECK (on_bp_medication IN (0,1)),
    on_statin               INTEGER CHECK (on_statin IN (0,1))
);

-- Current lab / vital snapshot per patient. Union of every LAB input the five
-- models consume. One row per patient here (the latest visit); modelled as a
-- history table (measured_at) so it could hold longitudinal labs later.
CREATE TABLE biomarkers (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id              TEXT NOT NULL REFERENCES demographics(patient_id),
    measured_at             TEXT NOT NULL,               -- ISO date/datetime
    systolic_bp             REAL,                        -- mmHg
    diastolic_bp            REAL,                        -- mmHg
    total_cholesterol_mgdl  REAL,                        -- mg/dL
    hdl_cholesterol_mgdl    REAL,                        -- mg/dL
    ldl_cholesterol_mgdl    REAL,                        -- mg/dL
    triglycerides_mgdl      REAL,                        -- mg/dL
    hba1c_percent           REAL,                        -- %
    fasting_glucose_mgdl    REAL,                        -- mg/dL
    egfr_ml_min_1_73m2      REAL,                        -- mL/min/1.73m^2
    creatinine_mgdl         REAL,                        -- mg/dL
    uacr_mg_g               REAL,                        -- urine albumin-creatinine ratio, mg/g
    urine_dipstick_protein  TEXT CHECK (urine_dipstick_protein IN ('negative','trace','1+','2+','3+')),
    ggt_u_l                 REAL,                        -- gamma-glutamyl transferase, U/L (CLivD)
    alt_u_l                 REAL,                        -- U/L
    ast_u_l                 REAL                         -- U/L
);
CREATE INDEX idx_biomarkers_patient ON biomarkers(patient_id, measured_at);

-- Append log of computed risk probabilities. The candidate's get_current_risks
-- endpoint computes today's values via the MLflow models and INSERTs a row per
-- risk here, so the assistant can show a trend over time.
CREATE TABLE risks (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id              TEXT NOT NULL REFERENCES demographics(patient_id),
    risk_code               TEXT NOT NULL CHECK (risk_code IN ('CVD','T2DM','CKD','CLD','DEMENTIA')),
    probability             REAL NOT NULL,               -- 0..1
    risk_band               TEXT CHECK (risk_band IN ('low','borderline','intermediate','high')),
    model_name              TEXT NOT NULL,
    model_version           TEXT,
    time_horizon_years      INTEGER,
    computed_at             TEXT NOT NULL,               -- ISO date/datetime
    inputs_json             TEXT                         -- payload snapshot (audit / citation-faithfulness)
);
CREATE INDEX idx_risks_patient ON risks(patient_id, risk_code, computed_at);
"""


# --- Demographics -----------------------------------------------------------
# Column order matches DEMOGRAPHICS_COLS below.
DEMOGRAPHICS_COLS = [
    "patient_id", "mrn", "first_name", "last_name", "date_of_birth", "sex",
    "height_cm", "weight_kg", "waist_cm", "hip_cm", "education_years",
    "smoking_status", "alcohol_drinks_per_week", "physical_activity_active",
    "family_history_diabetes", "hx_diabetes", "hx_hypertension",
    "gestational_diabetes", "on_bp_medication", "on_statin",
]

DEMOGRAPHICS = [
    # P001 — healthy young woman; all risks should be low.
    ("P001", "MRN-1001", "Maya",     "Cohen",    "1992-03-14", "female", 165, 60, 74, 98, 18, "never",   2,  1, 0, 0, 0, 0, 0, 0),
    # P002 — 68y male, high CVD: smoker, high chol/low HDL, hypertensive, overweight.
    ("P002", "MRN-1002", "David",    "Levi",     "1958-01-22", "male",   175, 92, 106, 102, 12, "current", 7, 0, 0, 0, 1, None, 1, 1),
    # P003 — 57y woman, high T2DM: obese, family hx, prediabetic, hypertensive, prior GDM.
    ("P003", "MRN-1003", "Sarah",    "Mizrahi",  "1969-05-02", "female", 162, 88, 104, 112, 14, "former",  3, 0, 1, 0, 1, 1, 1, 0),
    # P004 — 72y male, CKD + CVD: diabetic, hypertensive, reduced eGFR, proteinuria.
    ("P004", "MRN-1004", "Avraham",  "Friedman", "1954-02-18", "male",   172, 85, 102, 100, 16, "former",  4, 0, 1, 1, 1, None, 1, 1),
    # P005 — 61y male, CLD: heavy alcohol, very high GGT, high WHR, diabetic, smoker.
    ("P005", "MRN-1005", "Yosef",    "Katz",     "1965-04-11", "male",   178, 96, 112, 100, 12, "current", 28, 0, 0, 1, 1, None, 0, 0),
    # P006 — 66y woman, dementia lean: low education, hypertensive, high chol, obese, inactive.
    ("P006", "MRN-1006", "Rivka",    "Shapiro",  "1960-06-20", "female", 160, 82, 100, 110, 8,  "never",   1, 0, 0, 0, 1, 0, 1, 0),
    # P007 — 49y woman, borderline across the board.
    ("P007", "MRN-1007", "Noa",      "Bar",      "1977-02-10", "female", 168, 72, 86, 100, 15, "former",  5, 1, 1, 0, 0, 0, 0, 0),
    # P008 — 45y male, T2DM-leaning: overweight, family hx, sedentary, prehypertensive.
    ("P008", "MRN-1008", "Daniel",   "Green",    "1981-03-30", "male",   180, 98, 104, 102, 16, "never",   9, 0, 0, 0, 0, None, 0, 0),
]


# --- Biomarkers -------------------------------------------------------------
BIOMARKERS_COLS = [
    "patient_id", "measured_at", "systolic_bp", "diastolic_bp",
    "total_cholesterol_mgdl", "hdl_cholesterol_mgdl", "ldl_cholesterol_mgdl",
    "triglycerides_mgdl", "hba1c_percent", "fasting_glucose_mgdl",
    "egfr_ml_min_1_73m2", "creatinine_mgdl", "uacr_mg_g",
    "urine_dipstick_protein", "ggt_u_l", "alt_u_l", "ast_u_l",
]

BIOMARKERS = [
    # pid    sbp dbp  tchol hdl  ldl  tg   hba1c gluc egfr creat uacr dipstick    ggt alt ast
    ("P001", 112, 72, 178, 68, 95,  75,  5.1, 88,  102, 0.80, 5,  "negative", 18, 18, 20),
    ("P002", 158, 94, 255, 36, 175, 210, 5.8, 104, 74,  1.10, 18, "trace",    55, 32, 30),
    ("P003", 142, 88, 214, 44, 138, 180, 6.1, 116, 82,  0.90, 22, "negative", 34, 28, 24),
    ("P004", 150, 86, 190, 40, 118, 165, 7.4, 152, 52,  1.50, 85, "1+",       40, 26, 28),
    ("P005", 138, 84, 205, 38, 130, 240, 7.0, 140, 88,  0.95, 15, "negative", 145, 68, 74),
    ("P006", 156, 90, 262, 52, 175, 150, 5.6, 96,  79,  0.85, 12, "negative", 26, 20, 22),
    ("P007", 128, 82, 205, 55, 128, 120, 5.6, 98,  92,  0.82, 8,  "negative", 24, 22, 21),
    ("P008", 132, 85, 220, 42, 148, 190, 5.9, 108, 95,  0.90, 10, "negative", 38, 30, 26),
]


# --- Seeded risk history ----------------------------------------------------
# Back-dated rows so a trend exists before the candidate ever calls the endpoint.
# These probabilities are illustrative (plain data, NOT model outputs); they are
# filled in by tools/seed_risk_history in the author workflow to trend toward the
# current model-computed band. Columns: (patient_id, risk_code, computed_at,
# probability, risk_band, model_name, model_version, time_horizon_years).
# Populated after model calibration — see generate_models.py / verification step.
RISK_HISTORY: list[tuple] = [
    ("P001", "CVD", "2025-07-09", 0.036, "low", "prevent_cvd", "1.0.0", 10),
    ("P001", "CVD", "2026-01-09", 0.032, "low", "prevent_cvd", "1.0.0", 10),
    ("P001", "T2DM", "2025-07-09", 0.044, "low", "ada_t2dm", "1.0.0", None),
    ("P001", "T2DM", "2026-01-09", 0.038, "low", "ada_t2dm", "1.0.0", None),
    ("P001", "CKD", "2025-07-09", 0.024, "low", "framingham_ckd", "1.0.0", 10),
    ("P001", "CKD", "2026-01-09", 0.021, "low", "framingham_ckd", "1.0.0", 10),
    ("P001", "CLD", "2025-07-09", 0.026, "low", "clivd_cld", "1.0.0", 15),
    ("P001", "CLD", "2026-01-09", 0.023, "low", "clivd_cld", "1.0.0", 15),
    ("P001", "DEMENTIA", "2025-07-09", 0.038, "low", "caide_dementia", "1.0.0", 20),
    ("P001", "DEMENTIA", "2026-01-09", 0.033, "low", "caide_dementia", "1.0.0", 20),
    ("P002", "CVD", "2025-07-09", 0.58, "high", "prevent_cvd", "1.0.0", 10),
    ("P002", "CVD", "2026-01-09", 0.505, "high", "prevent_cvd", "1.0.0", 10),
    ("P002", "T2DM", "2025-07-09", 0.405, "high", "ada_t2dm", "1.0.0", None),
    ("P002", "T2DM", "2026-01-09", 0.353, "high", "ada_t2dm", "1.0.0", None),
    ("P002", "CKD", "2025-07-09", 0.323, "intermediate", "framingham_ckd", "1.0.0", 10),
    ("P002", "CKD", "2026-01-09", 0.282, "intermediate", "framingham_ckd", "1.0.0", 10),
    ("P002", "CLD", "2025-07-09", 0.163, "borderline", "clivd_cld", "1.0.0", 15),
    ("P002", "CLD", "2026-01-09", 0.142, "borderline", "clivd_cld", "1.0.0", 15),
    ("P002", "DEMENTIA", "2025-07-09", 0.518, "high", "caide_dementia", "1.0.0", 20),
    ("P002", "DEMENTIA", "2026-01-09", 0.451, "high", "caide_dementia", "1.0.0", 20),
    ("P003", "CVD", "2025-07-09", 0.217, "intermediate", "prevent_cvd", "1.0.0", 10),
    ("P003", "CVD", "2026-01-09", 0.189, "borderline", "prevent_cvd", "1.0.0", 10),
    ("P003", "T2DM", "2025-07-09", 0.743, "high", "ada_t2dm", "1.0.0", None),
    ("P003", "T2DM", "2026-01-09", 0.648, "high", "ada_t2dm", "1.0.0", None),
    ("P003", "CKD", "2025-07-09", 0.115, "borderline", "framingham_ckd", "1.0.0", 10),
    ("P003", "CKD", "2026-01-09", 0.1, "borderline", "framingham_ckd", "1.0.0", 10),
    ("P003", "CLD", "2025-07-09", 0.055, "low", "clivd_cld", "1.0.0", 15),
    ("P003", "CLD", "2026-01-09", 0.048, "low", "clivd_cld", "1.0.0", 15),
    ("P003", "DEMENTIA", "2025-07-09", 0.277, "intermediate", "caide_dementia", "1.0.0", 20),
    ("P003", "DEMENTIA", "2026-01-09", 0.241, "intermediate", "caide_dementia", "1.0.0", 20),
    ("P004", "CVD", "2025-07-09", 0.298, "intermediate", "prevent_cvd", "1.0.0", 10),
    ("P004", "CVD", "2026-01-09", 0.344, "intermediate", "prevent_cvd", "1.0.0", 10),
    ("P004", "T2DM", "2025-07-09", 0.338, "intermediate", "ada_t2dm", "1.0.0", None),
    ("P004", "T2DM", "2026-01-09", 0.391, "high", "ada_t2dm", "1.0.0", None),
    ("P004", "CKD", "2025-07-09", 0.39, "high", "framingham_ckd", "1.0.0", 10),
    ("P004", "CKD", "2026-01-09", 0.45, "high", "framingham_ckd", "1.0.0", 10),
    ("P004", "CLD", "2025-07-09", 0.078, "low", "clivd_cld", "1.0.0", 15),
    ("P004", "CLD", "2026-01-09", 0.09, "low", "clivd_cld", "1.0.0", 15),
    ("P004", "DEMENTIA", "2025-07-09", 0.209, "intermediate", "caide_dementia", "1.0.0", 20),
    ("P004", "DEMENTIA", "2026-01-09", 0.242, "intermediate", "caide_dementia", "1.0.0", 20),
    ("P005", "CVD", "2025-07-09", 0.251, "intermediate", "prevent_cvd", "1.0.0", 10),
    ("P005", "CVD", "2026-01-09", 0.29, "intermediate", "prevent_cvd", "1.0.0", 10),
    ("P005", "T2DM", "2025-07-09", 0.215, "intermediate", "ada_t2dm", "1.0.0", None),
    ("P005", "T2DM", "2026-01-09", 0.248, "intermediate", "ada_t2dm", "1.0.0", None),
    ("P005", "CKD", "2025-07-09", 0.108, "borderline", "framingham_ckd", "1.0.0", 10),
    ("P005", "CKD", "2026-01-09", 0.125, "borderline", "framingham_ckd", "1.0.0", 10),
    ("P005", "CLD", "2025-07-09", 0.312, "intermediate", "clivd_cld", "1.0.0", 15),
    ("P005", "CLD", "2026-01-09", 0.36, "high", "clivd_cld", "1.0.0", 15),
    ("P005", "DEMENTIA", "2025-07-09", 0.203, "intermediate", "caide_dementia", "1.0.0", 20),
    ("P005", "DEMENTIA", "2026-01-09", 0.234, "intermediate", "caide_dementia", "1.0.0", 20),
    ("P006", "CVD", "2025-07-09", 0.344, "intermediate", "prevent_cvd", "1.0.0", 10),
    ("P006", "CVD", "2026-01-09", 0.3, "intermediate", "prevent_cvd", "1.0.0", 10),
    ("P006", "T2DM", "2025-07-09", 0.423, "high", "ada_t2dm", "1.0.0", None),
    ("P006", "T2DM", "2026-01-09", 0.369, "high", "ada_t2dm", "1.0.0", None),
    ("P006", "CKD", "2025-07-09", 0.163, "borderline", "framingham_ckd", "1.0.0", 10),
    ("P006", "CKD", "2026-01-09", 0.142, "borderline", "framingham_ckd", "1.0.0", 10),
    ("P006", "CLD", "2025-07-09", 0.053, "low", "clivd_cld", "1.0.0", 15),
    ("P006", "CLD", "2026-01-09", 0.046, "low", "clivd_cld", "1.0.0", 15),
    ("P006", "DEMENTIA", "2025-07-09", 0.594, "high", "caide_dementia", "1.0.0", 20),
    ("P006", "DEMENTIA", "2026-01-09", 0.518, "high", "caide_dementia", "1.0.0", 20),
    ("P007", "CVD", "2025-07-09", 0.098, "low", "prevent_cvd", "1.0.0", 10),
    ("P007", "CVD", "2026-01-09", 0.085, "low", "prevent_cvd", "1.0.0", 10),
    ("P007", "T2DM", "2025-07-09", 0.148, "borderline", "ada_t2dm", "1.0.0", None),
    ("P007", "T2DM", "2026-01-09", 0.129, "borderline", "ada_t2dm", "1.0.0", None),
    ("P007", "CKD", "2025-07-09", 0.051, "low", "framingham_ckd", "1.0.0", 10),
    ("P007", "CKD", "2026-01-09", 0.044, "low", "framingham_ckd", "1.0.0", 10),
    ("P007", "CLD", "2025-07-09", 0.044, "low", "clivd_cld", "1.0.0", 15),
    ("P007", "CLD", "2026-01-09", 0.038, "low", "clivd_cld", "1.0.0", 15),
    ("P007", "DEMENTIA", "2025-07-09", 0.115, "borderline", "caide_dementia", "1.0.0", 20),
    ("P007", "DEMENTIA", "2026-01-09", 0.1, "borderline", "caide_dementia", "1.0.0", 20),
    ("P008", "CVD", "2025-07-09", 0.146, "borderline", "prevent_cvd", "1.0.0", 10),
    ("P008", "CVD", "2026-01-09", 0.127, "borderline", "prevent_cvd", "1.0.0", 10),
    ("P008", "T2DM", "2025-07-09", 0.19, "borderline", "ada_t2dm", "1.0.0", None),
    ("P008", "T2DM", "2026-01-09", 0.165, "borderline", "ada_t2dm", "1.0.0", None),
    ("P008", "CKD", "2025-07-09", 0.042, "low", "framingham_ckd", "1.0.0", 10),
    ("P008", "CKD", "2026-01-09", 0.036, "low", "framingham_ckd", "1.0.0", 10),
    ("P008", "CLD", "2025-07-09", 0.087, "low", "clivd_cld", "1.0.0", 15),
    ("P008", "CLD", "2026-01-09", 0.076, "low", "clivd_cld", "1.0.0", 15),
    ("P008", "DEMENTIA", "2025-07-09", 0.158, "borderline", "caide_dementia", "1.0.0", 20),
    ("P008", "DEMENTIA", "2026-01-09", 0.138, "borderline", "caide_dementia", "1.0.0", 20),
]


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            f"INSERT INTO demographics ({', '.join(DEMOGRAPHICS_COLS)}) "
            f"VALUES ({', '.join('?' for _ in DEMOGRAPHICS_COLS)})",
            DEMOGRAPHICS,
        )
        # BIOMARKERS rows carry no date; every snapshot is the same clinic visit.
        bio_cols = [c for c in BIOMARKERS_COLS if c != "measured_at"]
        conn.executemany(
            f"INSERT INTO biomarkers ({', '.join(bio_cols)}, measured_at) "
            f"VALUES ({', '.join(['?'] * len(bio_cols))}, ?)",
            [row + (LAST_VISIT,) for row in BIOMARKERS],
        )
        if RISK_HISTORY:
            conn.executemany(
                "INSERT INTO risks (patient_id, risk_code, computed_at, probability, "
                "risk_band, model_name, model_version, time_horizon_years) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                RISK_HISTORY,
            )
        conn.commit()

        n_dem = conn.execute("SELECT COUNT(*) FROM demographics").fetchone()[0]
        n_bio = conn.execute("SELECT COUNT(*) FROM biomarkers").fetchone()[0]
        n_rsk = conn.execute("SELECT COUNT(*) FROM risks").fetchone()[0]
        print(f"Wrote {DB_PATH}")
        print(f"  demographics: {n_dem} rows")
        print(f"  biomarkers:   {n_bio} rows")
        print(f"  risks:        {n_rsk} rows (seeded history)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
