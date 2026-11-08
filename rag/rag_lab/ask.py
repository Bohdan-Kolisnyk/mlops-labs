"""Ask one question over the archive.

    uv run --project rag python -m rag_lab.ask "What did he say about steel tariffs in spring 2025?"
"""

from __future__ import annotations

import argparse

from rag_lab import config, prompts
from rag_lab.embed import Embedder
from rag_lab.llm import Claude
from rag_lab.retrieval import answer_prompts, check_index, retrieve
from rag_lab.store import PgStore


def main() -> None:
    config.load_env()
    cfg = config.load()
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--top-k", type=int, default=cfg["retrieval"]["top_k"])
    args = ap.parse_args()

    store = PgStore(config.warehouse_dsn(), cfg["index"]["table"], cfg["embedding"]["dim"])
    check_index(store, cfg)
    hits = retrieve(store, Embedder(cfg["embedding"]["model"]), [args.question], args.top_k)
    template, phash, _ = prompts.load(cfg, "rag_answer")
    reply = Claude(cfg).ask(answer_prompts(template, [args.question], hits)[0])
    print(reply.text, "\n")
    print(f"sources (top {args.top_k}):")
    for h in hits[0]:
        print(f"  {h['post_id']}  score {h['score']:.3f}  {h['text'][:90]}")
    print(
        f"\nprompt {phash}, {reply.tokens_in} tokens in, {reply.tokens_out} out, "
        f"{reply.seconds:.2f}s, ${reply.cost:.5f}"
    )


if __name__ == "__main__":
    main()
