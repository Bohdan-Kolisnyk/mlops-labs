"""Readers for the two raw sources: the posts archive and the hourly index bars."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from mlops_labs import config


class SchemaError(ValueError):
    """The raw data does not have the agreed schema (design 6.8, contract with the source)."""


def check_posts_schema(path: str | Path) -> None:
    """Raise SchemaError if the posts file lacks a contract column or has a wrong type."""
    schema = pq.read_schema(path)
    missing = [c for c in config.RAW_POST_COLUMNS if c not in schema.names]
    if missing:
        raise SchemaError(f"posts file {Path(path).name} is missing columns: {missing}")
    for col in ("post_id", "datetime", "text"):
        if not (
            str(schema.field(col).type).startswith("string")
            or str(schema.field(col).type) == "large_string"
        ):
            raise SchemaError(f"column {col} must be a string, got {schema.field(col).type}")


def load_posts(path: str | Path) -> pd.DataFrame:
    """All posts, sorted by time. `created_at` is UTC; a missing text becomes ''."""
    df = pd.read_parquet(path, columns=config.RAW_POST_COLUMNS)
    df["created_at"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["text"] = df["text"].fillna("")
    df["has_media"] = df["has_media"].fillna(False).astype(bool)
    df = df.drop(columns=["datetime"])
    return df.sort_values("created_at", kind="stable").reset_index(drop=True)


def load_bars(path: str | Path) -> pd.DataFrame:
    """Hourly ^GSPC bars from the yfinance CSV (three header rows), sorted by start time."""
    df = pd.read_csv(
        path,
        skiprows=3,
        header=None,
        names=["ts", "adj_close", "close", "high", "low", "open", "volume"],
    )
    out = pd.DataFrame(
        {
            "bar_start": pd.to_datetime(df["ts"], utc=True),
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "volume": df["volume"],
        }
    )
    out = out.sort_values("bar_start").reset_index(drop=True)
    et = out["bar_start"].dt.tz_convert(config.ET)
    out["session_date"] = et.dt.normalize().dt.tz_localize(None)
    out["slot"] = et.dt.strftime("%H:%M")
    return out


def read_data_version(raw_dir: str | Path) -> str:
    """DVC hash of the raw directory, read from the `raw.dvc` file next to it.

    Falls back to an explicit marker when the directory is not the versioned one
    (for example the deliberately broken copy used to test the quality gate).
    """
    raw_dir = Path(raw_dir)
    dvc_file = raw_dir.parent / f"{raw_dir.name}.dvc"
    if dvc_file.exists():
        m = re.search(r"md5:\s*([0-9a-f]{32}(?:\.dir)?)", dvc_file.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    return f"unversioned:{raw_dir.name}"
