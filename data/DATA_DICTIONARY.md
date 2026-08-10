# Data Dictionary — `patient_db.db`

Mock clinical database for the assignment. One clinic, 8 fictional patients
(`P001`–`P008`). Regenerate with `uv run python data/generate_db.py`.

- **Clinic "today" is `2026-07-09`.** Derive age from `date_of_birth` against this
  fixed date so results are deterministic.
- `biomarkers` holds **one current snapshot per patient** (latest visit,
  `measured_at = 2026-06-15`), but is modelled as a history table.
- `risks` is an **append log**: it ships pre-seeded with two back-dated rows per
  patient per risk so a trend already exists; your `get_current_risks` endpoint
  appends today's freshly-computed row on top.

The **“Feeds”** column names which risk model(s) consume each field. Several model
inputs are **derived**, not stored — see [Derived quantities](#derived-quantities-not-stored).
Risk codes: **CVD** cardiovascular · **T2DM** type-2 diabetes · **CKD** chronic
kidney disease · **CLD** chronic liver disease · **DEM** dementia.

---

## Table: `demographics`
Identity, lifestyle, and medical-history flags. One row per patient.

| Column | Type | Units / values | Meaning | Feeds |
|---|---|---|---|---|
| `patient_id` | TEXT PK | `P001`… | Stable patient identifier | (key) |
| `mrn` | TEXT | `MRN-1001`… | Medical record number | — |
| `first_name`, `last_name` | TEXT | | Patient name | — |
| `date_of_birth` | TEXT | ISO `YYYY-MM-DD` | Birth date → derive **age** | all |
| `sex` | TEXT | `male` \| `female` | Biological sex → `sex_male` | CVD, T2DM, CLD, DEM |
| `height_cm` | REAL | cm | Height → **BMI** | CVD, T2DM, DEM |
| `weight_kg` | REAL | kg | Weight → **BMI** | CVD, T2DM, DEM |
| `waist_cm` | REAL | cm | Waist → **WHR** | CLD |
| `hip_cm` | REAL | cm | Hip → **WHR** | CLD |
| `education_years` | INT | years | Formal education | DEM |
| `smoking_status` | TEXT | `never`\|`former`\|`current` | → `current_smoker` flag | CVD, CLD |
| `alcohol_drinks_per_week` | REAL | drinks/week | Alcohol intake | CLD |
| `physical_activity_active` | INT | `0`\|`1` | 1 = physically active | T2DM, DEM |
| `family_history_diabetes` | INT | `0`\|`1` | First-degree family history | T2DM |
| `hx_diabetes` | INT | `0`\|`1` | Diagnosed diabetes | CVD, CKD, CLD |
| `hx_hypertension` | INT | `0`\|`1` | Diagnosed hypertension | T2DM, CKD |
| `gestational_diabetes` | INT | `0`\|`1`\|NULL | Prior GDM; **NULL where N/A** (males) | T2DM |
| `on_bp_medication` | INT | `0`\|`1` | On antihypertensive → `bp_treated` | CVD |
| `on_statin` | INT | `0`\|`1` | On a statin | CVD |

## Table: `biomarkers`
Latest labs and vitals. One row per patient.

| Column | Type | Units | Meaning | Feeds |
|---|---|---|---|---|
| `id` | INT PK | | Row id | — |
| `patient_id` | TEXT FK | | → `demographics.patient_id` | (key) |
| `measured_at` | TEXT | ISO date | Snapshot date | — |
| `systolic_bp` | REAL | mmHg | Systolic blood pressure | CVD, DEM |
| `diastolic_bp` | REAL | mmHg | Diastolic blood pressure | — |
| `total_cholesterol_mgdl` | REAL | mg/dL | Total cholesterol | CVD, DEM |
| `hdl_cholesterol_mgdl` | REAL | mg/dL | HDL cholesterol | CVD |
| `ldl_cholesterol_mgdl` | REAL | mg/dL | LDL cholesterol | — |
| `triglycerides_mgdl` | REAL | mg/dL | Triglycerides | — |
| `hba1c_percent` | REAL | % | Glycated haemoglobin | (context) |
| `fasting_glucose_mgdl` | REAL | mg/dL | Fasting glucose | (context) |
| `egfr_ml_min_1_73m2` | REAL | mL/min/1.73m² | Estimated GFR (kidney function) | CVD, CKD |
| `creatinine_mgdl` | REAL | mg/dL | Serum creatinine | — |
| `uacr_mg_g` | REAL | mg/g | Urine albumin-creatinine ratio | (context) |
| `urine_dipstick_protein` | TEXT | `negative`\|`trace`\|`1+`\|`2+`\|`3+` | → `proteinuria_trace_plus` flag | CKD |
| `ggt_u_l` | REAL | U/L | Gamma-glutamyl transferase | CLD |
| `alt_u_l` | REAL | U/L | Alanine aminotransferase | — |
| `ast_u_l` | REAL | U/L | Aspartate aminotransferase | — |

Columns marked *(context)* are clinically relevant and available to show the
clinician, but are **not** inputs to the surrogate models in this assignment.

## Table: `risks`
Append log of computed risk probabilities (seeded history + your live rows).

| Column | Type | Meaning |
|---|---|---|
| `id` | INT PK | Row id |
| `patient_id` | TEXT FK | → `demographics.patient_id` |
| `risk_code` | TEXT | `CVD`\|`T2DM`\|`CKD`\|`CLD`\|`DEMENTIA` |
| `probability` | REAL | Model output, 0–1 |
| `risk_band` | TEXT | `low`\|`borderline`\|`intermediate`\|`high` (see thresholds) |
| `model_name` | TEXT | e.g. `prevent_cvd` |
| `model_version` | TEXT | e.g. `1.0.0` |
| `time_horizon_years` | INT | 10 / 15 / 20; NULL for T2DM screening |
| `computed_at` | TEXT | ISO date/datetime of computation |
| `inputs_json` | TEXT | Payload snapshot used (audit / citation-faithfulness) |

---

## Derived quantities (not stored)
These are **model inputs you must compute** from the columns above — this is part
of building the payload:

| Derived input | From | Formula / rule |
|---|---|---|
| `age_years` | `date_of_birth` | whole years at clinic-today `2026-07-09` |
| `bmi` | `weight_kg`, `height_cm` | `weight_kg / (height_cm/100)²` |
| `waist_hip_ratio` | `waist_cm`, `hip_cm` | `waist_cm / hip_cm` |
| `sex_male` | `sex` | `1` if `male` else `0` |
| `current_smoker` | `smoking_status` | `1` if `current` else `0` |
| `proteinuria_trace_plus` | `urine_dipstick_protein` | `1` if not `negative` else `0` |
| `bp_treated` | `on_bp_medication` | pass through `0/1` |

> The **exact feature names, order, and encodings each model expects** are not
> listed here on purpose — discover them from the model itself
> (`model.feature_names_in_`) as described in [`models/README.md`](../models/README.md).

## Risk banding thresholds
Illustrative, uniform across risks (real instruments band per-outcome):

| Band | Probability |
|---|---|
| `low` | `< 0.10` |
| `borderline` | `0.10 – < 0.20` |
| `intermediate` | `0.20 – < 0.35` |
| `high` | `≥ 0.35` |

## Designed risk profile (sanity reference)
Each patient was constructed to have a clear headline risk; use this to sanity-check
your pipeline end-to-end:

| Patient | Headline | Note |
|---|---|---|
| P001 Maya Cohen | none | healthy 34F — everything low |
| P002 David Levi | **CVD** | 68M smoker, high chol/low HDL, hypertensive |
| P003 Sarah Mizrahi | **T2DM** | obese, family hx, prediabetic, prior GDM |
| P004 Avraham Friedman | **CKD** | diabetic, eGFR 52, proteinuria (also high CVD) |
| P005 Yosef Katz | **CLD** | heavy alcohol, GGT 145, high WHR |
| P006 Rivka Shapiro | **Dementia** | low education, hypertensive, high chol, obese |
| P007 Noa Bar | borderline | 49F, mixed borderline signals |
| P008 Daniel Green | T2DM (mod) | 45M overweight, family hx, sedentary |
