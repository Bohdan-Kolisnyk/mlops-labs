.PHONY: setup lint test up down data etl-month

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
