"""Build the pgvector index once: do nothing when it already matches config.yaml and the corpus.

    python -m rag_lab.index_job

Runs as a Kubernetes Job next to the cluster's PostgreSQL. Waits for the database, compares the
stored config hash with the one computed from config.yaml and the corpus file, and runs the Lab 3
ingestion (`rag_lab.ingest`) only when they differ.
"""

from __future__ import annotations

import time

from rag_lab import config, ingest
from rag_lab.store import PgStore


def wait_for_database(store: PgStore, timeout_s: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            store.index_config_hash()
            return
        except Exception as exc:
            message = str(exc)
            if "does not exist" in message and "rag_index_meta" in message:
                return
            if time.monotonic() > deadline:
                raise RuntimeError(f"the database did not answer: {exc}") from exc
            print(f"waiting for the database: {type(exc).__name__}", flush=True)
            time.sleep(3)


def main() -> None:
    config.load_env()
    cfg = config.load()
    store = PgStore(config.warehouse_dsn(), cfg["index"]["table"], cfg["embedding"]["dim"])
    wait_for_database(store)
    try:
        built = store.index_config_hash()
    except Exception:  # noqa: BLE001
        built = None
    if built == config.index_hash(cfg):
        documents, chunks = store.count()
        print(f"index already built: {documents} posts, {chunks} chunks, config {built}", flush=True)
        return
    ingest.main()


if __name__ == "__main__":
    main()
