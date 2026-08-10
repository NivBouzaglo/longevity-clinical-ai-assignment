# Convenience targets. Each long-running service wants its own terminal.
# Ports: backend 8001 · MLflow 5001 · MCP 9000 · LibreChat 3080 (Docker).

.PHONY: help install data db models backend mcp mlflow test eval lint

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install dependencies
	uv sync

data: db models  ## Regenerate the mock DB and the models

db:  ## Regenerate data/patient_db.db
	uv run python data/generate_db.py

models:  ## Regenerate models/*.pkl
	uv run python models/generate_models.py

backend:  ## Run the FastAPI backend on :8001 (implement the endpoints first)
	uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8001 --reload

mcp:  ## Run the FastMCP server on 0.0.0.0:9000
	uv run python mcp-server/server.py

mlflow:  ## Reminder: serve YOUR registered risk model on :5001 (see GUIDE.md §4)
	@echo "Register + serve your model, e.g.:"
	@echo "  uv run mlflow models serve -m models/mlflow_risk_router -p 5001 --env-manager local"

test:  ## Run the backend tests
	uv run pytest

eval:  ## Run your evaluation harness (you build evals/harness.py)
	uv run python evals/harness.py

lint:  ## Lint with ruff
	uv run ruff check .
