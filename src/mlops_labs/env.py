"""Tiny .env reader, so scripts on the host see the same settings as docker compose."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_env(path: str | Path | None = None) -> None:
    """Put KEY=VALUE lines of .env into os.environ without overriding what is already set."""
    p = Path(path) if path else ROOT / ".env"
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
