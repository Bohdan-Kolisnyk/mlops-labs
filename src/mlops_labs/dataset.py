"""The training table: one row per usable post, with text, features, label and split.

Two sources give the same table (a test checks it):
  warehouse - clean_posts joined with features, as written by the Airflow DAG;
  raw       - computed straight from data/raw with the same functions the DAG calls. This
              is what the reproduction command uses, because it needs only the DVC data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from mlops_labs import config
from mlops_labs.data import load_bars, load_posts, read_data_version
from mlops_labs.features import FEATURE_COLUMNS, compute_features
from mlops_labs.labels import build_windows, label_posts

TABLE_COLUMNS = [
    "post_id",
    "created_at",
    "bar_start",
    "session_date",
    "text",
    "y",
    *FEATURE_COLUMNS,
]


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df[TABLE_COLUMNS].sort_values(["created_at", "post_id"], kind="stable")
    return df.reset_index(drop=True)


def from_raw(raw_dir: str | Path) -> tuple[pd.DataFrame, str]:
    """(training table, data version) computed from the raw directory."""
    raw_dir = Path(raw_dir)
    posts = load_posts(raw_dir / config.POSTS_FILE)
    bars = load_bars(raw_dir / config.BARS_FILE)
    labelled = label_posts(posts, build_windows(bars))
    labelled["session_date"] = pd.to_datetime(labelled["session_date"])
    feats = compute_features(labelled, posts, bars)
    feats = feats.drop(columns=["event_timestamp", "feature_version"])
    df = feats.merge(
        labelled[["post_id", "created_at", "bar_start", "session_date", "text", "y"]],
        on="post_id",
        how="inner",
        validate="one_to_one",
    )
    return _finish(df), read_data_version(raw_dir)


def from_warehouse(engine: Engine) -> tuple[pd.DataFrame, str]:
    """(training table, data version) read from the warehouse tables."""
    cols = ", ".join(f"f.{c}" for c in FEATURE_COLUMNS)
    query = text(
        f"SELECT f.post_id, f.feature_version, {cols}, "
        "c.created_at, c.bar_start, c.session_date, c.text, c.y, c.data_version "
        "FROM features f JOIN clean_posts c ON c.post_id = f.post_id AND c.batch = f.batch"
    )
    df = pd.read_sql(query, engine)
    if df.empty:
        raise RuntimeError("the features table is empty: run the ETL DAG first")
    versions = sorted(df["data_version"].unique())
    if len(versions) != 1:
        raise RuntimeError(f"warehouse mixes several data versions: {versions}")
    if set(df["feature_version"]) != {config.FEATURE_VERSION}:
        raise RuntimeError(f"unexpected feature versions: {sorted(set(df['feature_version']))}")
    for col in ("created_at", "bar_start"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df["session_date"] = pd.to_datetime(df["session_date"])
    return _finish(df), versions[0]


def split_name(session_date: pd.Series) -> pd.Series:
    """'train' / 'val' / 'test' by the post's session date (design 5.4); never random."""
    d = session_date
    out = pd.Series("", index=d.index, dtype="object")
    out[d <= pd.Timestamp(config.TRAIN_END)] = "train"
    out[(d >= pd.Timestamp(config.VAL_START)) & (d <= pd.Timestamp(config.VAL_END))] = "val"
    out[d >= pd.Timestamp(config.TEST_START)] = "test"
    return out
