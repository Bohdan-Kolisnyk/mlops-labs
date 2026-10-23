"""Loading into PostgreSQL. The batch is replaced as a whole, so a rerun is harmless."""

from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Connection, Engine

TABLES = {"raw_posts", "clean_posts", "features"}


def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or os.environ["WAREHOUSE_URL"])


def replace_batch(conn: Connection, table: str, df: pd.DataFrame, batch: str) -> int:
    """Delete-then-insert for one batch inside the caller's transaction.

    Running this twice for the same batch leaves exactly the same rows, because the
    DELETE removes whatever an earlier run of the batch inserted.
    """
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}")
    if inspect(conn).has_table(table):
        conn.execute(text(f"DELETE FROM {table} WHERE batch = :batch"), {"batch": batch})
    df.to_sql(table, conn, if_exists="append", index=False, chunksize=1000)
    return len(df)


def count_batch(conn: Connection, table: str, batch: str) -> int:
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}")
    if not inspect(conn).has_table(table):
        return 0
    row = conn.execute(
        text(f"SELECT COUNT(*) FROM {table} WHERE batch = :batch"), {"batch": batch}
    ).scalar_one()
    return int(row)
