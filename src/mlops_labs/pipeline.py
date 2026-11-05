"""The steps of the monthly ETL DAG, as plain functions (the DAG only calls them).

Batch = one calendar month of posts, taken from the DAG run's data interval or from the
`month` parameter, never from the wall clock.

extract -> validate -> transform -> quality_check -> load -> build_features

The tasks hand data to each other through the warehouse, never through the orchestrator:
`extract` and `transform` write the scratch tables `stg_posts` and `stg_clean` (one batch at
a time, replaced on every run), `quality_check` reads `stg_clean`, and only `load` copies
the batch into `raw_posts` and `clean_posts`. A task returns just a few counters (metadata).
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from mlops_labs import config
from mlops_labs.data import (
    SchemaError,
    check_posts_schema,
    load_bars,
    load_posts,
    read_data_version,
)
from mlops_labs.features import compute_features
from mlops_labs.labels import build_windows, label_posts
from mlops_labs.quality import assert_quality
from mlops_labs.warehouse import TABLES, count_batch, replace_batch

log = logging.getLogger(__name__)


def batch_bounds(
    month: str | None,
    data_interval_start: pd.Timestamp | None = None,
    data_interval_end: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """[start, end) in UTC and the batch name 'YYYY-MM'.

    The `month` parameter wins when given (manual run for one month); otherwise the
    DAG run's data interval is used (scheduled run).
    """
    if month:
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
    elif data_interval_start is not None and data_interval_end is not None:
        start = pd.Timestamp(data_interval_start).tz_convert("UTC")
        end = pd.Timestamp(data_interval_end).tz_convert("UTC")
        if end <= start:  # Airflow 3 without an explicit timetable: start == end
            end = start + pd.offsets.MonthBegin(1)
    else:
        raise ValueError("neither a month parameter nor a data interval was given")
    return start, end, start.strftime("%Y-%m")


STAGE_POSTS = "stg_posts"
STAGE_CLEAN = "stg_clean"


def _stage_write(engine: Engine, table: str, df: pd.DataFrame, batch: str) -> None:
    """Replace the batch in a scratch table (same delete-then-insert as the final tables)."""
    df = df.copy()
    df["batch"] = batch
    with engine.begin() as conn:
        replace_batch(conn, table, df, batch)


def read_batch(engine: Engine, table: str, batch: str) -> pd.DataFrame:
    """The rows of one batch from a warehouse table, with proper types."""
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}")
    df = pd.read_sql(text(f"SELECT * FROM {table} WHERE batch = :b"), engine, params={"b": batch})
    for col in ("created_at", "bar_start"):
        if col in df:
            df[col] = pd.to_datetime(df[col], utc=True)
    if "session_date" in df:
        df["session_date"] = pd.to_datetime(df["session_date"])
    for col in ("text_empty", "dup_text", "has_media"):
        if col in df:
            df[col] = df[col].astype(bool)
    return df


def extract(raw_dir: str | Path, engine: Engine, start, end, batch: str) -> dict:
    """Read the posts of the batch from the raw directory into the scratch table stg_posts."""
    posts = load_posts(Path(raw_dir) / config.POSTS_FILE)
    t = posts["created_at"]
    rows = posts[(t >= start) & (t < end)]
    _stage_write(engine, STAGE_POSTS, rows, batch)
    log.info("extract %s: %d of %d posts", batch, len(rows), len(posts))
    return {"batch": batch, "posts_in_batch": len(rows), "posts_total": len(posts)}


def validate(raw_dir: str | Path, engine: Engine, batch: str) -> dict:
    """Schema and basic sanity checks on the raw batch, before any transformation."""
    check_posts_schema(Path(raw_dir) / config.POSTS_FILE)
    rows = read_batch(engine, STAGE_POSTS, batch)
    if rows.empty:
        raise SchemaError(f"batch {batch} has no posts")
    problems = []
    if rows["post_id"].isna().any() or rows["created_at"].isna().any():
        problems.append("null post_id or unparsable timestamp")
    if rows["post_id"].duplicated().any():
        problems.append("duplicated post_id")
    if problems:
        raise SchemaError(f"batch {batch}: " + ", ".join(problems))
    return {"batch": batch, "rows": len(rows)}


def transform(raw_dir: str | Path, engine: Engine, start, end, batch: str) -> dict:
    """Label the posts, flag empty and duplicate texts, keep the posts of the batch.

    The labels and the duplicate flags are computed on the whole archive and then cut to
    the batch, so they depend only on the raw data version and not on the order of runs.
    """
    raw_dir = Path(raw_dir)
    posts = load_posts(raw_dir / config.POSTS_FILE)
    bars = load_bars(raw_dir / config.BARS_FILE)
    labelled = label_posts(posts, build_windows(bars))
    t = labelled["created_at"]
    clean = labelled[(t >= start) & (t < end)].copy()
    clean["session_date"] = pd.to_datetime(clean["session_date"])
    clean["data_version"] = read_data_version(raw_dir)
    clean["batch"] = batch
    clean = clean[
        [
            "post_id",
            "created_at",
            "text",
            "has_media",
            "bar_start",
            "session_date",
            "slot",
            "r",
            "sigma",
            "y",
            "text_empty",
            "dup_text",
            "data_version",
            "batch",
        ]
    ]
    _stage_write(engine, STAGE_CLEAN, clean, batch)
    stats = {
        "batch": batch,
        "clean_rows": len(clean),
        "empty_text_rows": int(clean["text_empty"].sum()),
        "duplicate_text_rows": int(clean["dup_text"].sum()),
    }
    log.info("transform %s: %s", batch, stats)
    return stats


def load(engine: Engine, batch: str) -> dict:
    """Replace the batch in raw_posts and clean_posts, in one transaction."""
    raw = read_batch(engine, STAGE_POSTS, batch)
    clean = read_batch(engine, STAGE_CLEAN, batch)
    with engine.begin() as conn:
        replace_batch(conn, "raw_posts", raw, batch)
        replace_batch(conn, "clean_posts", clean, batch)
    with engine.connect() as conn:
        counts = {
            "raw_posts": count_batch(conn, "raw_posts", batch),
            "clean_posts": count_batch(conn, "clean_posts", batch),
        }
    log.info("load %s: row counts %s", batch, counts)
    return counts


def read_clean_batch(engine: Engine, batch: str) -> pd.DataFrame:
    """The loaded clean rows of one batch, with proper types."""
    return read_batch(engine, "clean_posts", batch)


def quality_check(raw_dir: str | Path, engine: Engine, start, end, batch: str) -> dict:
    """The quality gate, between transform and load: a bad batch never reaches the warehouse.

    Raises QualityGateError when any check fails.
    """
    clean = read_batch(engine, STAGE_CLEAN, batch)
    bars = load_bars(Path(raw_dir) / config.BARS_FILE)
    results = assert_quality(clean, bars, start, end)
    for r in results:
        log.info("quality %s: %s - %s", batch, "ok" if r.passed else "FAILED", r.detail)
    return {r.name: r.detail for r in results}


def build_features(engine: Engine, raw_dir: str | Path, batch: str) -> dict:
    """Fill the feature table for the batch (only posts that passed the cleaning rules)."""
    raw_dir = Path(raw_dir)
    clean = read_clean_batch(engine, batch)
    posts = load_posts(raw_dir / config.POSTS_FILE)
    bars = load_bars(raw_dir / config.BARS_FILE)
    feats = compute_features(clean, posts, bars)
    feats["batch"] = batch
    with engine.begin() as conn:
        replace_batch(conn, "features", feats, batch)
    with engine.connect() as conn:
        n = count_batch(conn, "features", batch)
    log.info("build_features %s: %d rows, feature_version=%s", batch, n, config.FEATURE_VERSION)
    return {"batch": batch, "feature_rows": n, "feature_version": config.FEATURE_VERSION}


def materialize_online(feast_python: str | Path, feature_repo: str | Path) -> dict:
    """Publish the newest sigma of each slot to the Feast online store (PostgreSQL).

    Feast has its own virtualenv (it pins pandas below 3), so the work is done by
    feature_repo/materialize.py run with that interpreter. It prints one JSON line and fails
    when a slot has no value online; that line is what the task returns.
    """
    feature_repo = Path(feature_repo)
    done = subprocess.run(
        [str(feast_python), str(feature_repo / "materialize.py")],
        cwd=feature_repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
    if done.returncode != 0 or not lines:
        raise RuntimeError(f"Feast materialize failed:\n{done.stdout}\n{done.stderr}")
    result = json.loads(lines[-1])
    log.info("materialize: %s", result)
    return result
