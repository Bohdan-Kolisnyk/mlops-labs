"""The two lineage identifiers of Lab 2: the data version and the code commit."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# What a training run depends on. Edits here make the commit tag untrue.
CODE_PATHS = ["src", "scripts", "pyproject.toml", "uv.lock"]


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, encoding="utf-8"
    )
    return out.stdout.strip()


def git_commit() -> str:
    """Full hash of HEAD, or 'unknown' outside a git checkout."""
    try:
        return _git("rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty() -> bool:
    """True when code the run depends on has uncommitted changes (tracked or new files)."""
    try:
        return bool(_git("status", "--porcelain", "--", *CODE_PATHS))
    except (OSError, subprocess.CalledProcessError):
        return True
