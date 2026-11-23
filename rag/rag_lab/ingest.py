"""Corpus -> chunks -> embeddings -> pgvector index. One command, prints the document count.

    uv run --project rag python -m rag_lab.ingest

Reads rag/data/corpus.parquet (built from data/raw by scripts/export_rag_corpus.py --corpus-only when it is missing) and every setting from
rag/config.yaml. The index table is dropped and rebuilt, so the same corpus and the same config
always give the same index; the config hash is stored with it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pandas as pd

from rag_lab import config
from rag_lab.core import build_chunks
from rag_lab.embed import Embedder
from rag_lab.store import PgStore


def ensure_corpus(cfg: dict) -> None:
    target = config.path_of(cfg["corpus"]["path"])
    if target.exists():
        return
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit(f"{target} is missing and uv is not on PATH: run scripts/export_rag_corpus.py --corpus-only")
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    cmd = [uv, "run", "--project", str(config.ROOT), "python", "scripts/export_rag_corpus.py", "--corpus-only"]
    print(f"{target.name} is missing, building it from data/raw: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=config.ROOT, env=env, check=True)


def main() -> None:
    config.load_env()
    cfg = config.load()
    ensure_corpus(cfg)
    corpus = pd.read_parquet(config.path_of(cfg["corpus"]["path"]))
    ch, emb = cfg["chunking"], cfg["embedding"]
    chunks = build_chunks(corpus, ch["size_words"], ch["overlap_words"], cfg["corpus"]["min_chars"])
    vectors = Embedder(emb["model"]).encode(chunks["text"].tolist())
    store = PgStore(config.warehouse_dsn(), cfg["index"]["table"], emb["dim"])
    store.rebuild(chunks, vectors, config.index_hash(cfg))
    posts, n_chunks = store.count()
    manifest = {
        "documents": posts,
        "chunks": n_chunks,
        "config_hash": config.index_hash(cfg),
        "embedding_model": emb["model"],
        "chunk_size_words": ch["size_words"],
        "overlap_words": ch["overlap_words"],
    }
    out = config.path_of("data/index_manifest.json")
    out.write_text(json.dumps(manifest, indent=2))
    print(f"indexed {posts} documents (posts) as {n_chunks} chunks, config {manifest['config_hash']}")


if __name__ == "__main__":
    main()
