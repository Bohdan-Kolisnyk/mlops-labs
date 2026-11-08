"""Pure functions: chunking, retrieval metrics, latency percentiles, question checks."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Windows of `size` words, each starting `size - overlap` words after the previous one."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than the chunk size")
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return out


def build_chunks(corpus: pd.DataFrame, size: int, overlap: int, min_chars: int) -> pd.DataFrame:
    """One row per chunk. The chunk text starts with the post date, so a question about a
    period has something to match."""
    rows = []
    for r in corpus.itertuples(index=False):
        if len(r.text.strip()) < min_chars:
            continue
        for n, piece in enumerate(chunk_words(r.text, size, overlap)):
            rows.append(
                {
                    "chunk_id": f"{r.post_id}#{n}",
                    "post_id": r.post_id,
                    "chunk_no": n,
                    "text": f"[{str(r.created_at)[:10]}] {piece}",
                    "created_at": r.created_at,
                    "y": int(r.y),
                    "ret": float(r.ret),
                }
            )
    return pd.DataFrame(rows)


def unique_posts(chunk_hits: list[dict]) -> list[str]:
    """Post ids of the ranked chunks, best first, each post once."""
    seen, out = set(), []
    for h in chunk_hits:
        if h["post_id"] not in seen:
            seen.add(h["post_id"])
            out.append(h["post_id"])
    return out


def hit_rate(ranked: list[list[str]], relevant: list[set[str]], k: int) -> float:
    """Share of questions with at least one relevant post among the first k."""
    hits = [bool(set(r[:k]) & rel) for r, rel in zip(ranked, relevant, strict=True)]
    return float(np.mean(hits))


def mrr(ranked: list[list[str]], relevant: list[set[str]], k: int) -> float:
    """Mean reciprocal rank of the first relevant post within the first k (0 when absent)."""
    scores = []
    for r, rel in zip(ranked, relevant, strict=True):
        rr = 0.0
        for i, pid in enumerate(r[:k], start=1):
            if pid in rel:
                rr = 1.0 / i
                break
        scores.append(rr)
    return float(np.mean(scores))


def percentiles(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95))}


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = re.findall(r"[\w']+", text.lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def quotes_the_post(question: str, post_text: str, n: int = 4) -> bool:
    """True when the question shares a run of n words with the post: such a question is
    answered by string overlap and tells nothing about the retriever."""
    return bool(_ngrams(question, n) & _ngrams(post_text, n))
