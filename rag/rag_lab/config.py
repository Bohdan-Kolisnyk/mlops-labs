"""One configuration file for ingestion, retrieval, the LLM calls and the evaluation runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

RAG_DIR = Path(__file__).resolve().parents[1]
ROOT = RAG_DIR.parent
CONFIG_PATH = RAG_DIR / "config.yaml"


def load_env(path: Path | None = None) -> None:
    """KEY=VALUE lines of .env into os.environ (what is already set wins)."""
    p = path or ROOT / ".env"
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load(path: Path | None = None) -> dict:
    return yaml.safe_load((path or CONFIG_PATH).read_text(encoding="utf-8"))


def path_of(cfg_path: str) -> Path:
    return RAG_DIR / cfg_path


def index_hash(cfg: dict) -> str:
    """Hash of everything that decides what the index contains. The index stores it, the query
    path compares it: a different chunk size or embedding model than the one indexed, or a
    corpus whose posts changed since the index was built, is refused."""
    parts = {
        "chunking": cfg["chunking"],
        "embedding": cfg["embedding"],
        "min_chars": cfg["corpus"]["min_chars"],
        "corpus": corpus_hash(path_of(cfg["corpus"]["path"])),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def corpus_hash(path: Path) -> str:
    """Hash of the post ids and texts of the corpus file (not of the parquet bytes), or "none"."""
    if not path.exists():
        return "none"
    import pandas as pd

    df = pd.read_parquet(path, columns=["post_id", "text"]).sort_values("post_id")
    digest = hashlib.sha256()
    for post_id, text in zip(df["post_id"], df["text"], strict=True):
        digest.update(f"{post_id}\x1f{text}\x1e".encode())
    return digest.hexdigest()[:12]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def warehouse_dsn() -> str:
    """WAREHOUSE_URL is a SQLAlchemy URL; psycopg wants a plain libpq one."""
    url = os.environ["WAREHOUSE_URL"]
    return url.replace("postgresql+psycopg2://", "postgresql://")
