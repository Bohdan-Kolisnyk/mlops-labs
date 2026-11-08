"""The query path: question -> top-k chunks -> prompt -> answer."""

from __future__ import annotations

from rag_lab import prompts
from rag_lab.config import index_hash


def check_index(store, cfg: dict) -> None:
    """Refuse to answer from an index built with other settings than the ones in config.yaml."""
    built, want = store.index_config_hash(), index_hash(cfg)
    if built != want:
        raise RuntimeError(
            f"the index was built with config {built} but config.yaml is {want}: "
            "rebuild it with `python -m rag_lab.ingest`"
        )


def retrieve(store, embedder, questions: list[str], k: int) -> list[list[dict]]:
    vectors = embedder.encode(questions)
    return [store.search(v, k) for v in vectors]


def context_of(hits: list[dict]) -> str:
    lines = []
    for h in hits:
        reaction = f"abnormal move: {'yes' if h['y'] else 'no'}, S&P 500 two-bar return {h['ret']:+.2%}"
        lines.append(f"- {h['text']} (market reaction: {reaction})")
    return "\n".join(lines)


def answer_prompts(template: str, questions: list[str], hits: list[list[dict]]) -> list[str]:
    return [
        prompts.render(template, question=q, context=context_of(h))
        for q, h in zip(questions, hits, strict=True)
    ]
