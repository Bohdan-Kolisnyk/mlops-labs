"""Monthly ETL for the Truth Social alert system (Lab 2).

extract -> validate -> transform -> quality_check -> load -> build_features -> materialize

Batch = one calendar month. A scheduled run takes it from the run's data interval; a manual
run takes it from the `month` parameter (for example {"month": "2025-03"}). The current
time is never used, so rerunning or back-filling a month gives the same rows.

Airflow 3 does not create data intervals for a plain "@monthly" schedule (start == end), so
the interval is declared explicitly with CronDataIntervalTimetable.

The tasks do not pass data to each other. Each one works out its batch from the run context,
reads its input from the warehouse (scratch tables stg_posts and stg_clean) or from the raw
files, writes its output there, and returns only a few counters.

The last task, materialize, publishes the newest sigma of each time slot from the Feast
offline store (parquet in MinIO) to the online store (PostgreSQL), which scripts/predict.py
reads with --online. The order is declared with
`>>`, not by feeding one task's result into the next.
"""

from __future__ import annotations

import os

import pendulum
from airflow.sdk import DAG, Param, get_current_context, task
from airflow.timetables.interval import CronDataIntervalTimetable

RAW_DIR = os.environ.get("RAW_DATA_DIR", "/opt/airflow/data/raw")
FEAST_PYTHON = os.environ.get("FEAST_PYTHON", "/home/airflow/feast-venv/bin/python")
FEAST_REPO = os.environ.get("FEAST_REPO", "/opt/airflow/feature_repo")

with DAG(
    dag_id="truth_social_etl",
    schedule=CronDataIntervalTimetable("@monthly", timezone="UTC"),
    start_date=pendulum.datetime(2023, 12, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    params={
        "month": Param(None, type=["null", "string"], description="YYYY-MM, manual runs"),
        "raw_dir": Param(RAW_DIR, type="string", description="directory with the raw files"),
    },
    tags=["lab2", "etl"],
    doc_md=__doc__,
) as dag:

    def _batch() -> dict:
        """Resolve the batch from the run context (imported lazily: heavy imports)."""
        from mlops_labs.pipeline import batch_bounds

        ctx = get_current_context()
        start, end, name = batch_bounds(
            ctx["params"].get("month"),
            ctx.get("data_interval_start"),
            ctx.get("data_interval_end"),
        )
        return {
            "start": start,
            "end": end,
            "batch": name,
            "raw_dir": ctx["params"].get("raw_dir") or RAW_DIR,
        }

    @task
    def extract() -> dict:
        """Posts of the batch from the raw files into the scratch table stg_posts."""
        from mlops_labs import pipeline
        from mlops_labs.warehouse import get_engine

        b = _batch()
        return pipeline.extract(b["raw_dir"], get_engine(), b["start"], b["end"], b["batch"])

    @task
    def validate() -> dict:
        """Schema and sanity checks on the extracted posts."""
        from mlops_labs import pipeline
        from mlops_labs.warehouse import get_engine

        b = _batch()
        return pipeline.validate(b["raw_dir"], get_engine(), b["batch"])

    @task
    def transform() -> dict:
        """Labels and cleaning flags into the scratch table stg_clean."""
        from mlops_labs import pipeline
        from mlops_labs.warehouse import get_engine

        b = _batch()
        return pipeline.transform(b["raw_dir"], get_engine(), b["start"], b["end"], b["batch"])

    @task
    def quality_check() -> dict:
        """The quality gate. Raises on a bad batch, so nothing is loaded and no features built."""
        from mlops_labs import pipeline
        from mlops_labs.warehouse import get_engine

        b = _batch()
        return pipeline.quality_check(b["raw_dir"], get_engine(), b["start"], b["end"], b["batch"])

    @task
    def load() -> dict:
        """Replace the batch in the warehouse (delete-then-insert in one transaction)."""
        from mlops_labs import pipeline
        from mlops_labs.warehouse import get_engine

        return pipeline.load(get_engine(), _batch()["batch"])

    @task
    def build_features() -> dict:
        from mlops_labs import pipeline
        from mlops_labs.warehouse import get_engine

        b = _batch()
        return pipeline.build_features(get_engine(), b["raw_dir"], b["batch"])

    @task
    def materialize() -> dict:
        """Newest sigma of every slot from the offline store into the online store (Feast)."""
        from mlops_labs import pipeline

        return pipeline.materialize_online(FEAST_PYTHON, FEAST_REPO)

    (
        extract()
        >> validate()
        >> transform()
        >> quality_check()
        >> load()
        >> build_features()
        >> materialize()
    )
