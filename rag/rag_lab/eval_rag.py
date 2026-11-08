"""Evaluate retrieval and answers on the evaluation set, one MLflow run per top-k value.

    uv run --project rag python -m rag_lab.eval_rag                    # the sweep in config.yaml
    uv run --project rag python -m rag_lab.eval_rag --top-k 5 --no-answers

Retrieval metrics (hit rate at k, MRR) are computed here from the retrieved post ids. The answer
metric is an LLM-as-judge score (1-5 against the reference, divided by 5) with a versioned judge
prompt. Every run logs the settings, both prompt hashes, tokens, cost and p50/p95 latency.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from rag_lab import config, prompts, track
from rag_lab.core import hit_rate, mrr, unique_posts
from rag_lab.embed import Embedder
from rag_lab.llm import Claude, ask_many, usage
from rag_lab.retrieval import answer_prompts, check_index, retrieve
from rag_lab.store import PgStore


def load_questions(cfg: dict) -> list[dict]:
    path = config.path_of(cfg["eval"]["set_path"])
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evaluate(cfg, questions, store, embedder, llm, k: int, answers: bool = True) -> dict:
    """Metrics and per-question results for one top-k value."""
    qs = [q["question"] for q in questions]
    hits = retrieve(store, embedder, qs, k)
    ranked = [unique_posts(h) for h in hits]
    relevant = [set(q["relevant_post_ids"]) for q in questions]
    metrics = {"hit_rate": hit_rate(ranked, relevant, k), "mrr": mrr(ranked, relevant, k)}
    rows = [
        {"qid": q["qid"], "question": q["question"], "relevant": q["relevant_post_ids"],
         "retrieved": r, "hit": bool(set(r[:k]) & set(q["relevant_post_ids"]))}
        for q, r in zip(questions, ranked, strict=True)
    ]
    if not answers:
        return {"metrics": metrics, "rows": rows}

    workers = cfg["llm"]["concurrency"]
    a_tpl, _, _ = prompts.load(cfg, "rag_answer")
    j_tpl, _, _ = prompts.load(cfg, "judge")
    a_replies = ask_many(llm, answer_prompts(a_tpl, qs, hits), workers)
    judge_in = [
        prompts.render(j_tpl, question=q["question"], reference=q["reference_answer"], answer=a.text)
        for q, a in zip(questions, a_replies, strict=True)
    ]
    j_replies = ask_many(llm, judge_in, workers)
    scores = []
    for row, a, j in zip(rows, a_replies, j_replies, strict=True):
        obj = prompts.parse_json(j.text) or {}
        s = obj.get("score")
        row.update(answer=a.text, judge_score=s, judge_reason=obj.get("reason"))
        if isinstance(s, (int, float)) and 1 <= s <= 5:
            scores.append(float(s))
    a_use, j_use = usage(a_replies), usage(j_replies)
    metrics.update(
        answer_score=float(np.mean(scores)) / 5 if scores else float("nan"),
        judge_parse_failures=len(rows) - len(scores),
        answer_calls=a_use["calls"],
        answer_tokens_in=a_use["tokens_in"],
        answer_tokens_out=a_use["tokens_out"],
        answer_latency_p50_s=a_use["latency_p50_s"],
        answer_latency_p95_s=a_use["latency_p95_s"],
        judge_tokens_in=j_use["tokens_in"],
        judge_tokens_out=j_use["tokens_out"],
        cost_usd=a_use["cost_usd"] + j_use["cost_usd"],
        cost_per_question_usd=(a_use["cost_usd"] + j_use["cost_usd"]) / len(rows),
    )
    return {"metrics": metrics, "rows": rows}


def main() -> None:
    config.load_env()
    cfg = config.load()
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, nargs="+", default=cfg["retrieval"]["sweep_top_k"])
    ap.add_argument("--no-answers", action="store_true", help="retrieval metrics only, no LLM")
    args = ap.parse_args()

    track.setup(cfg)
    questions = load_questions(cfg)
    store = PgStore(config.warehouse_dsn(), cfg["index"]["table"], cfg["embedding"]["dim"])
    check_index(store, cfg)
    embedder = Embedder(cfg["embedding"]["model"])
    llm = None if args.no_answers else Claude(cfg)
    _, a_hash, a_path = prompts.load(cfg, "rag_answer")
    _, j_hash, j_path = prompts.load(cfg, "judge")
    set_path = config.path_of(cfg["eval"]["set_path"])
    documents, chunks = store.count()
    runs_dir = config.path_of("data/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    for k in args.top_k:
        res = evaluate(cfg, questions, store, embedder, llm, k, answers=not args.no_answers)
        out = runs_dir / f"eval_top{k}.jsonl"
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in res["rows"]), encoding="utf-8")
        params = {
            "top_k": k,
            "embedding_model": cfg["embedding"]["model"],
            "chunk_size_words": cfg["chunking"]["size_words"],
            "overlap_words": cfg["chunking"]["overlap_words"],
            "index_config_hash": config.index_hash(cfg),
            "index_documents": documents,
            "index_chunks": chunks,
            "n_questions": len(questions),
            "eval_set_hash": config.file_hash(set_path),
            "llm_model": cfg["llm"]["model"],
            "temperature": "model default",
            "max_tokens": cfg["llm"]["max_tokens"],
            "answer_prompt_hash": a_hash,
            "judge_prompt_hash": j_hash,
            "answers": not args.no_answers,
        }
        run_id = track.log_run(
            cfg["mlflow"]["rag_experiment"],
            f"rag-top{k}" + ("-retrieval-only" if args.no_answers else ""),
            params,
            res["metrics"],
            files=[a_path, j_path, set_path, out],
            tags={"lab": "3", "kind": "rag-eval"},
        )
        m = res["metrics"]
        print(f"top-k {k}: hit rate {m['hit_rate']:.3f}, MRR {m['mrr']:.3f}"
              + ("" if args.no_answers else f", answer score {m['answer_score']:.3f}, "
                 f"cost ${m['cost_usd']:.4f}, p50 {m['answer_latency_p50_s']:.2f}s, "
                 f"p95 {m['answer_latency_p95_s']:.2f}s") + f"  (run {run_id[:8]})")


if __name__ == "__main__":
    main()
