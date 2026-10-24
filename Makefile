.PHONY: setup lint test up down data etl-month months train reproduce features feast-demo

setup:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest

up:
	docker compose up -d --build

down:
	docker compose down

data:
	uv run dvc pull

# make etl-month MONTH=2025-03
etl-month:
	docker compose exec -T airflow-scheduler airflow dags trigger truth_social_etl --conf '{"month": "$(MONTH)"}'

# all 35 months, one DAG run each (they queue and run one after another)
months:
	uv run python scripts/trigger_months.py

# needs the stack up and all months loaded; commit first (the commit is a lineage tag)
train:
	uv run python -m mlops_labs.train --register

# rebuild the registered champion from its tags and compare the metrics
reproduce:
	uv run python scripts/reproduce.py models:/truth-social-abnormal-move@champion

# Feast runs in its own environment (feature_repo/pyproject.toml): it pins pandas below 3
features:
	uv run python scripts/export_slot_sigma.py

feast-demo: features
	cd feature_repo && uv sync && uv run feast apply && uv run python demo.py
