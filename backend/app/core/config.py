"""Application settings, loaded from environment / the repo-root .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated vars (MCP_BEARER_TOKEN, JWT_SECRET, ...)
    )

    app_name: str = "Longevity Clinical AI — Backend"

    # Path to the mock patient SQLite database (env: PATIENT_DB_PATH).
    patient_db_path: Path = REPO_ROOT / "data" / "patient_db.db"

    # MLflow model server /invocations endpoint (env: MLFLOW_URL).
    # Default assumes you serve the router model on host port 5001.
    mlflow_url: str = "http://127.0.0.1:5001/invocations"


settings = Settings()
