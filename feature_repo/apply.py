"""Create the `feast` database if needed and register the feature definitions (feast apply).

cd feature_repo && uv run python apply.py
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
from definitions import slot, slot_sigma
from feast import FeatureStore
from feast_env import setup_env

HERE = Path(__file__).resolve().parent


def ensure_database() -> None:
    with psycopg.connect(
        host=os.environ["FEAST_DB_HOST"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname="postgres",
        autocommit=True,
    ) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = 'feast'").fetchone():
            conn.execute("CREATE DATABASE feast")
            print("created database feast")


def main() -> None:
    setup_env()
    ensure_database()
    store = FeatureStore(repo_path=str(HERE))
    store.apply([slot, slot_sigma])
    print("registered:", [e.name for e in store.list_entities()], end=" ")
    print([v.name for v in store.list_feature_views()])


if __name__ == "__main__":
    main()
