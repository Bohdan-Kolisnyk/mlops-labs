"""The query path: question -> top-k chunks -> prompt -> answer."""

from __future__ import annotations

from rag_lab import prompts
from rag_lab.config import index_hash
from rag_lab.sparse import BM25, rrf


def check_index(store, cfg: dict) -> None:
    """Refuse to answer from an index built with other settings than the ones in config.yaml."""
    built, want = store.index_config_hash(), index_hash(cfg)
    if built != want:
        raise RuntimeError(
            f"the index was built with config {built} but config.yaml is {want}: "
            "rebuild it with `python -m rag_lab.ingest`"
        )


def retrieve(
    store, embedder, questions: list[str], k: int, method: str = "dense", sparse=None, hybrid: dict | None = None
) -> list[list[dict]]:
    """Top-k chunks per question: dense (pgvector), bm25 (keywords) or hybrid (rank fusion of both)."""
    if method == "dense":
        return [store.search(v, k) for v in embedder.encode(questions)]
    if method == "bm25":
        return [sparse.search(q, k) for q in questions]
    if method == "hybrid":
        pool, rrf_k = hybrid["pool"], hybrid["rrf_k"]
        vectors = embedder.encode(questions)
        return [
            rrf([store.search(v, pool), sparse.search(q, pool)], k, rrf_k)
            for q, v in zip(questions, vectors, strict=True)
        ]
    raise ValueError(f"unknown retrieval method {method!r}")


def sparse_index(store, cfg: dict, method: str):
    """The BM25 index over the stored chunks, built only when the method needs it."""
    return BM25.from_store(store, cfg["bm25"]) if method in ("bm25", "hybrid") else None


def context_of(hits: list[dict]) -> str:
    lines = []
    for h in hits:
        reaction = f"abnormal move: {'yes' if h['y'] else 'no'}, S&P 500 two-bar return {h['ret']:+.2%}"
        lines.append(f"- {h['text']} (market reaction: {reaction})")
    return "\n".join(lines)


def numbered_context(hits: list[dict]) -> str:
    """The passages as the answer prompt shows them, numbered so a judge can point at one."""
    lines = []
    for i, h in enumerate(hits, start=1):
        reaction = f"abnormal move: {'yes' if h['y'] else 'no'}, S&P 500 two-bar return {h['ret']:+.2%}"
        lines.append(f"[{i}] {h['text']} (market reaction: {reaction})")
    return "\n".join(lines)


def answer_prompts(template: str, questions: list[str], hits: list[list[dict]]) -> list[str]:
    return [
        prompts.render(template, question=q, context=context_of(h))
        for q, h in zip(questions, hits, strict=True)
    ]
