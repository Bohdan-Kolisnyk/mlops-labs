"""Vector index: pgvector in the warehouse PostgreSQL, and an in-memory twin for tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MemoryStore:
    """Same interface as PgStore, brute-force cosine similarity (vectors are normalised)."""

    def __init__(self):
        self.chunks: pd.DataFrame | None = None
        self.vectors: np.ndarray | None = None
        self.config_hash: str | None = None

    def rebuild(self, chunks: pd.DataFrame, vectors: np.ndarray, config_hash: str) -> None:
        self.chunks, self.vectors, self.config_hash = chunks.reset_index(drop=True), vectors, config_hash

    def search(self, vector: np.ndarray, k: int) -> list[dict]:
        scores = self.vectors @ vector
        top = np.argsort(-scores)[:k]
        return [
            {**self.chunks.iloc[i].to_dict(), "score": float(scores[i])} for i in top
        ]

    def index_config_hash(self) -> str | None:
        return self.config_hash

    def count(self) -> tuple[int, int]:
        return int(self.chunks["post_id"].nunique()), len(self.chunks)


class PgStore:
    def __init__(self, dsn: str, table: str, dim: int):
        self.dsn, self.table, self.dim = dsn, table, dim

    def _connect(self):
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self.dsn, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return conn

    def rebuild(self, chunks: pd.DataFrame, vectors: np.ndarray, config_hash: str) -> None:
        """Drop and rebuild the whole index: the same corpus and config give the same index."""
        t = self.table
        with self._connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
            conn.execute(
                f"CREATE TABLE {t} (chunk_id text PRIMARY KEY, post_id text NOT NULL, "
                f"chunk_no int NOT NULL, text text NOT NULL, created_at timestamptz, "
                f"y int, ret double precision, embedding vector({self.dim}) NOT NULL)"
            )
            rows = [
                (r.chunk_id, r.post_id, int(r.chunk_no), r.text, r.created_at, int(r.y),
                 float(r.ret), vectors[i])
                for i, r in enumerate(chunks.itertuples(index=False))
            ]
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {t} (chunk_id, post_id, chunk_no, text, created_at, y, ret, "
                    f"embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    rows,
                )
            conn.execute(f"CREATE INDEX ON {t} USING hnsw (embedding vector_cosine_ops)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS rag_index_meta "
                "(table_name text PRIMARY KEY, config_hash text, built_at timestamptz DEFAULT now())"
            )
            conn.execute(
                "INSERT INTO rag_index_meta (table_name, config_hash) VALUES (%s, %s) "
                "ON CONFLICT (table_name) DO UPDATE SET config_hash = EXCLUDED.config_hash, "
                "built_at = now()",
                (t, config_hash),
            )

    def search(self, vector: np.ndarray, k: int) -> list[dict]:
        with self._connect() as conn:
            cur = conn.execute(
                f"SELECT chunk_id, post_id, chunk_no, text, created_at, y, ret, "
                f"1 - (embedding <=> %s) AS score FROM {self.table} "
                f"ORDER BY embedding <=> %s LIMIT %s",
                (vector, vector, k),
            )
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def index_config_hash(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT config_hash FROM rag_index_meta WHERE table_name = %s", (self.table,)
            ).fetchone()
        return row[0] if row else None

    def count(self) -> tuple[int, int]:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT count(DISTINCT post_id), count(*) FROM {self.table}"
            ).fetchone()
        return int(row[0]), int(row[1])
