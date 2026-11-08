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
    path compares it: a different chunk size or embedding model than the one indexed is refused."""
    parts = {
        "chunking": cfg["chunking"],
        "embedding": cfg["embedding"],
        "min_chars": cfg["corpus"]["min_chars"],
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def warehouse_dsn() -> str:
    """WAREHOUSE_URL is a SQLAlchemy URL; psycopg wants a plain libpq one."""
    url = os.environ["WAREHOUSE_URL"]
    return url.replace("postgresql+psycopg2://", "postgresql://")
