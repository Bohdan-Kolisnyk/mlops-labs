"""Monthly ETL for the Truth Social alert system (Lab 2).

extract -> validate -> transform -> quality_check -> load -> build_features

Batch = one calendar month. A scheduled run takes it from the run's data interval; a manual
run takes it from the `month` parameter (for example {"month": "2025-03"}). The current
time is never used, so rerunning or back-filling a month gives the same rows.

Airflow 3 does not create data intervals for a plain "@monthly" schedule (start == end), so
the interval is declared explicitly with CronDataIntervalTimetable.
"""

from __future__ import annotations

import os

import pendulum
from airflow.sdk import DAG, Param, get_current_context, task
from airflow.timetables.interval import CronDataIntervalTimetable

RAW_DIR = os.environ.get("RAW_DATA_DIR", "/opt/airflow/data/raw")
STAGING_DIR = os.environ.get("STAGING_DIR", "/opt/airflow/data/staging")

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
        from mlops_labs import pipeline

        b = _batch()
        info = pipeline.extract(b["raw_dir"], STAGING_DIR, b["start"], b["end"], b["batch"])
        return {"batch": b["batch"], "raw_dir": b["raw_dir"], **info}

    @task
    def validate(x: dict) -> dict:
        from mlops_labs import pipeline

        pipeline.validate(x["raw_dir"], STAGING_DIR, x["batch"])
        return x

    @task
    def transform(x: dict) -> dict:
        from mlops_labs import pipeline

        b = _batch()
        stats = pipeline.transform(x["raw_dir"], STAGING_DIR, b["start"], b["end"], x["batch"])
        return {**x, **stats}

    @task
    def quality_check(x: dict) -> dict:
        """The quality gate. Raises on a bad batch, so nothing is loaded and no features built."""
        from mlops_labs import pipeline

        b = _batch()
        pipeline.quality_check(x["raw_dir"], STAGING_DIR, b["start"], b["end"], x["batch"])
        return x

    @task
    def load(x: dict) -> dict:
        """Replace the batch in the warehouse (delete-then-insert in one transaction)."""
        from mlops_labs import pipeline
        from mlops_labs.warehouse import get_engine

        counts = pipeline.load(get_engine(), STAGING_DIR, x["batch"])
        return {**x, **counts}

    @task
    def build_features(x: dict) -> dict:
        from mlops_labs import pipeline
        from mlops_labs.warehouse import get_engine

        return pipeline.build_features(get_engine(), x["raw_dir"], x["batch"])

    build_features(load(quality_check(transform(validate(extract())))))
