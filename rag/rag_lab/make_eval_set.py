"""Generate the evaluation questions with the LLM, then READ them before you trust them.

    uv run --project rag python -m rag_lab.make_eval_set

For each sampled post the LLM writes a question that the post answers, asked by properties and
not by names, plus a short reference answer. A question that shares a run of four words with its
post is rejected in code (it would be answered by string overlap). Writes eval/eval_set.jsonl,
which is committed: the set must be rerunnable after every change.
"""

from __future__ import annotations

import json

import pandas as pd

from rag_lab import config, prompts
from rag_lab.core import quotes_the_post
from rag_lab.llm import Claude, ask_many


def main() -> None:
    config.load_env()
    cfg = config.load()
    corpus = pd.read_parquet(config.path_of(cfg["corpus"]["path"]))
    ok = corpus[corpus["text"].str.len().between(150, 1500)]
    sample = ok.sample(cfg["eval"]["candidates"], random_state=cfg["eval"]["seed"])
    template, phash, _ = prompts.load(cfg, "gen_question")
    texts = [
        prompts.render(template, date=str(r.created_at)[:10], text=r.text)
        for r in sample.itertuples(index=False)
    ]
    replies = ask_many(Claude(cfg), texts, cfg["llm"]["concurrency"])

    kept, rejected = [], 0
    for r, reply in zip(sample.itertuples(index=False), replies, strict=True):
        obj = prompts.parse_json(reply.text)
        if not obj or not obj.get("question") or not obj.get("answer"):
            rejected += 1
            continue
        if quotes_the_post(obj["question"], r.text):
            rejected += 1
            continue
        kept.append(
            {
                "qid": f"q{len(kept) + 1:02d}",
                "question": obj["question"].strip(),
                "relevant_post_ids": [r.post_id],
                "reference_answer": obj["answer"].strip(),
                "generated_by": cfg["llm"]["model"],
                "gen_prompt_hash": phash,
            }
        )
    out = config.path_of(cfg["eval"]["set_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(k, ensure_ascii=False) for k in kept) + "\n", encoding="utf-8")
    print(f"{len(kept)} questions kept, {rejected} rejected -> {out}")
    if len(kept) < 30:
        raise SystemExit(f"only {len(kept)} questions kept, the lab needs at least 30: raise eval.candidates")
    print("Read them now: python -c \"print(open('rag/eval/eval_set.jsonl', encoding='utf-8').read())\"")


if __name__ == "__main__":
    main()
