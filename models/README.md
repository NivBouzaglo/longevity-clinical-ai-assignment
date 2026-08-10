# Risk Models (`models/*.pkl`)

Five **provided** risk models, one per clinical risk. Each is a plain
scikit-learn `LogisticRegression` with a `predict_proba` method and
discoverable input feature names. Regenerate with
`uv run python models/generate_models.py`.

> ⚠️ **These are surrogates**, not clinically validated instruments. Coefficients
> are hand-set to be directionally correct (e.g. CVD risk rises with age, systolic
> BP, smoking) and calibrated to a plausible probability range. They exist to
> demonstrate real-time, biomarker-driven inference — nothing more.

| File | `risk_code` | Outcome | Horizon | Surrogate of |
|---|---|---|---|---|
| `prevent_cvd.pkl` | CVD | Cardiovascular disease | 10 yr | AHA PREVENT (2023) |
| `ada_t2dm.pkl` | T2DM | Type-2 diabetes (screening) | — | ADA Diabetes Risk Test |
| `framingham_ckd.pkl` | CKD | Chronic kidney disease (stage 3+) | 10 yr | Framingham CKD score |
| `clivd_cld.pkl` | CLD | Chronic liver disease | 15 yr | CLivD score (with GGT) |
| `caide_dementia.pkl` | DEMENTIA | Dementia | 20 yr | CAIDE score |

## Output semantics
Binary classifier over `classes_ = [0, 1]`. The **risk probability is the
positive-class column**: `model.predict_proba(X)[:, 1]` → a float in `(0, 1)`.

## Inspect a model (this is your source of truth)
Your job is to turn a patient's record into the feature vector each model expects.
Start by asking the model what it wants:

```python
import pickle
m = pickle.load(open("models/prevent_cvd.pkl", "rb"))
print(list(m.feature_names_in_))   # exact input names, in order
print(m.n_features_in_)            # how many
print(m.metadata_)                 # {'risk_code','model_name','outcome','time_horizon_years',...}
```

Pass inputs as a **pandas DataFrame whose columns match `feature_names_in_`
(names and order)** — that keeps MLflow happy and avoids the sklearn
"X does not have valid feature names" warning.

## Input contracts
The features each model consumes, with the **units/encoding the model assumes**.
All binary flags are `0/1`. `age_years` is whole years; `bmi` is kg/m²;
`waist_hip_ratio` is a plain ratio.

- **prevent_cvd** — `age_years`, `sex_male`, `total_cholesterol_mgdl` (mg/dL),
  `hdl_cholesterol_mgdl` (mg/dL), `systolic_bp` (mmHg), `bp_treated`, `on_statin`,
  `diabetes`, `current_smoker`, `bmi`, `egfr` (mL/min/1.73m²)
- **ada_t2dm** — `age_years`, `sex_male`, `bmi`, `family_history_diabetes`,
  `hypertension`, `physically_active`, `gestational_diabetes`
- **framingham_ckd** — `age_years`, `diabetes`, `hypertension`,
  `proteinuria_trace_plus`, `egfr` (mL/min/1.73m²)
- **clivd_cld** — `age_years`, `sex_male`, `alcohol_drinks_per_week`,
  `waist_hip_ratio`, `diabetes`, `current_smoker`, `ggt_u_l` (U/L)
- **caide_dementia** — `age_years`, `sex_male`, `education_years`, `systolic_bp`
  (mmHg), `bmi`, `total_cholesterol_mgdl` (mg/dL), `physically_active`

> **What you must figure out yourself:** how each of these maps to columns in
> `patient_db.db` (the names differ — e.g. the models want `diabetes`/`hypertension`
> flags, `bp_treated`, `current_smoker`, `proteinuria_trace_plus`), plus the derived
> values (`age_years`, `bmi`, `waist_hip_ratio`). Units and derivations are in
> [`data/DATA_DICTIONARY.md`](../data/DATA_DICTIONARY.md). Getting units and the
> feature order right is exactly what "numeric faithfulness" checks.

## Serving these with MLflow
You will serve these models behind MLflow and call them from the backend. One
important gotcha: MLflow's default `pyfunc` predict calls `.predict()` (class
labels), **not** `.predict_proba()`. You must arrange for probabilities to come
out. See [`GUIDE.md`](../GUIDE.md#4-serve-the-models-mlflow) for the recommended
single-router approach, the exact `/invocations` payload shape, and a `curl` smoke
test.
