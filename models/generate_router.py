"""Build and save the MLflow "router" pyfunc model that serves all five risk
models behind a single :5001 endpoint (see GUIDE.md §4).

The gotcha this exists to fix: MLflow's default pyfunc `predict` wrapping a
sklearn estimator calls `.predict()` (class *labels*, 0/1), not
`.predict_proba()`. A naive `mlflow models serve` on any one of the five
pickles would silently return labels and break the whole risk story.

``RiskRouter`` loads all five pickles once and, given a `model` param, routes
to the right one and returns `predict_proba(X)[:, 1]` — the positive-class
probability.

Run (regenerates the saved model directory, deterministic):
    uv run python models/generate_router.py

Serve it:
    uv run mlflow models serve -m models/mlflow_risk_router -p 5001 --env-manager local

Call it (see GUIDE.md §4 for the full curl example):
    POST http://127.0.0.1:5001/invocations
    {"dataframe_split": {"columns": [...], "data": [[...]]}, "params": {"model": "framingham_ckd"}}
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema

MODELS_DIR = Path(__file__).resolve().parent
ROUTER_PATH = MODELS_DIR / "mlflow_risk_router"

MODEL_FILES = {
    "prevent_cvd": MODELS_DIR / "prevent_cvd.pkl",
    "ada_t2dm": MODELS_DIR / "ada_t2dm.pkl",
    "framingham_ckd": MODELS_DIR / "framingham_ckd.pkl",
    "clivd_cld": MODELS_DIR / "clivd_cld.pkl",
    "caide_dementia": MODELS_DIR / "caide_dementia.pkl",
}


class RiskRouter(mlflow.pyfunc.PythonModel):
    """Dispatches a scoring request to one of five sklearn risk models by name.

    Always returns predict_proba(X)[:, 1] — the positive-class probability —
    never sklearn's .predict() class labels.
    """

    def load_context(self, context) -> None:
        self.models = {
            name: pickle.load(open(path, "rb"))  # noqa: SIM115
            for name, path in context.artifacts.items()
        }

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None):
        name = (params or {}).get("model")
        if name not in self.models:
            raise ValueError(f"Unknown model '{name}'. Known models: {sorted(self.models)}")
        model = self.models[name]
        # Column order matters to sklearn — select in the model's own declared order.
        X = pd.DataFrame(model_input)[list(model.feature_names_in_)]
        return model.predict_proba(X)[:, 1]


def _build_signature() -> ModelSignature:
    """Union of all five models' feature columns, each marked NOT required, so a
    caller can send just the subset of columns the chosen model actually needs
    (e.g. the 5-column CKD-only payload in GUIDE.md §4)."""
    all_columns: set[str] = set()
    for path in MODEL_FILES.values():
        m = pickle.load(open(path, "rb"))  # noqa: SIM115
        all_columns.update(m.feature_names_in_)

    input_schema = Schema(
        [ColSpec(DataType.double, name=col, required=False) for col in sorted(all_columns)]
    )
    output_schema = Schema([ColSpec(DataType.double)])
    param_schema = ParamSchema([ParamSpec("model", DataType.string, default="prevent_cvd")])
    return ModelSignature(inputs=input_schema, outputs=output_schema, params=param_schema)


def main() -> None:
    if ROUTER_PATH.exists():
        shutil.rmtree(ROUTER_PATH)

    signature = _build_signature()
    mlflow.pyfunc.save_model(
        path=str(ROUTER_PATH),
        python_model=RiskRouter(),
        artifacts={name: str(path) for name, path in MODEL_FILES.items()},
        signature=signature,
    )
    print(f"Saved router model -> {ROUTER_PATH}")
    print("Serve it with:")
    print(f"  uv run mlflow models serve -m {ROUTER_PATH} -p 5001 --env-manager local")


if __name__ == "__main__":
    main()
