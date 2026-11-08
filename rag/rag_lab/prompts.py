"""Prompts are files. Their hash goes into every run that used them."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rag_lab.config import file_hash, path_of


def load(cfg: dict, name: str) -> tuple[str, str, Path]:
    """(text, hash, path) of the prompt called `name` in the configuration."""
    p = path_of(cfg["prompts"][name])
    return p.read_text(encoding="utf-8"), file_hash(p), p


def render(template: str, **values: object) -> str:
    out = template
    for key, value in values.items():
        out = out.replace("{{" + key + "}}", str(value))
    return out


def parse_json(text: str) -> dict | None:
    """The first JSON object in a model reply, or None."""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
