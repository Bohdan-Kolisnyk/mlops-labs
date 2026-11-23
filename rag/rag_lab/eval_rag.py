"""Evaluate retrieval and answers on the evaluation set, one MLflow run per top-k value.

    uv run --project rag python -m rag_lab.eval_rag                    # the sweep in config.yaml
    uv run --project rag python -m rag_lab.eval_rag --top-k 5 --no-answers

Retrieval metrics (hit rate at k, MRR) are computed here from the retrieved post ids. The answer
metric is an LLM-as-judge score (1-5 against the reference, divided by 5) with a versioned judge
prompt. Every run logs the settings, both prompt hashes, tokens, cost and p50/p95 latency.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

from rag_lab import config, prompts, quality, track
from rag_lab.core import hit_rate, mrr, unique_posts
from rag_lab.embed import Embedder
from rag_lab.llm import Claude, ask_many, make_llm, model_name, usage
from rag_lab.retrieval import answer_prompts, check_index, retrieve, sparse_index
from rag_lab.store import PgStore


def load_questions(cfg: dict) -> list[dict]:
    path = config.path_of(cfg["eval"]["set_path"])
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evaluate(
    cfg, questions, store, embedder, llm, k: int, answers: bool = True, with_quality: bool = True,
    method: str = "dense", sparse=None, judge=None,
) -> dict:
    """Metrics and per-question results for one top-k value."""
    qs = [q["question"] for q in questions]
    hits = retrieve(store, embedder, qs, k, method, sparse, cfg.get("hybrid"))
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

    judge = judge or llm
    workers = cfg["llm"]["concurrency"]
    a_tpl, _, _ = prompts.load(cfg, "rag_answer")
    j_tpl, _, _ = prompts.load(cfg, "judge")
    started = time.perf_counter()
    a_workers = cfg["llm_local"]["concurrency"] if getattr(llm, "gpu_usd_per_hour", None) else workers
    a_replies = ask_many(llm, answer_prompts(a_tpl, qs, hits), a_workers)
    answer_wall = time.perf_counter() - started
    judge_in = [
        prompts.render(j_tpl, question=q["question"], reference=q["reference_answer"], answer=a.text)
        for q, a in zip(questions, a_replies, strict=True)
    ]
    j_replies = ask_many(judge, judge_in, workers)
    scores = []
    for row, a, j in zip(rows, a_replies, j_replies, strict=True):
        obj = prompts.parse_json(j.text) or {}
        s = obj.get("score")
        row.update(answer=a.text, stop_reason=a.stop_reason, judge_score=s, judge_reason=obj.get("reason"))
        if isinstance(s, (int, float)) and 1 <= s <= 5:
            scores.append(float(s))
    if with_quality:
        q_res = quality.judge_quality(cfg, judge, questions, [a.text for a in a_replies], hits)
        for row, extra in zip(rows, q_res["rows"], strict=True):
            row.update(extra)
        metrics.update(q_res["metrics"])
        hit_rows = [r for r in rows if r["hit"]]
        metrics["false_decline_rate"] = (
            sum(bool(r.get("declines")) for r in hit_rows) / len(hit_rows) if hit_rows else float("nan")
        )
        missed = [r for r in rows if not r["hit"]]
        metrics["decline_rate_when_gold_missing"] = (
            sum(bool(r.get("declines")) for r in missed) / len(missed) if missed else float("nan")
        )
    a_use, j_use = usage(a_replies), usage(j_replies)
    shadow = getattr(llm, "gpu_usd_per_hour", None)
    answer_cost = answer_wall * shadow / 3600 if shadow else a_use["cost_usd"]
    metrics.update(
        answer_score=float(np.mean(scores)) / 5 if scores else float("nan"),
        judge_parse_failures=len(rows) - len(scores),
        answers_not_finished=sum(a.stop_reason != "end_turn" for a in a_replies),
        answer_calls=a_use["calls"],
        answer_tokens_in=a_use["tokens_in"],
        answer_tokens_out=a_use["tokens_out"],
        answer_latency_p50_s=a_use["latency_p50_s"],
        answer_latency_p95_s=a_use["latency_p95_s"],
        judge_tokens_in=j_use["tokens_in"],
        judge_tokens_out=j_use["tokens_out"],
        answer_latency_max_s=a_use["latency_max_s"],
        answer_first_call_s=a_replies[0].seconds,
        answer_wall_s=answer_wall,
        answer_cost_usd=answer_cost,
        cost_usd=answer_cost + j_use["cost_usd"],
        cost_per_question_usd=(answer_cost + j_use["cost_usd"]) / len(rows),
    )
    return {"metrics": metrics, "rows": rows}


def main() -> None:
    config.load_env()
    cfg = config.load()
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, nargs="+", default=cfg["retrieval"]["sweep_top_k"])
    ap.add_argument("--method", nargs="+", choices=["dense", "bm25", "hybrid"],
                    default=cfg["retrieval"]["sweep_methods"])
    ap.add_argument("--answerer", choices=["anthropic", "local"], default="anthropic",
                    help="the model that writes the answers; the judges are always the Anthropic model")
    ap.add_argument("--no-answers", action="store_true", help="retrieval metrics only, no LLM")
    ap.add_argument("--no-quality", action="store_true", help="skip the groundedness, relevance and completeness judges")
    args = ap.parse_args()

    track.setup(cfg)
    questions = load_questions(cfg)
    store = PgStore(config.warehouse_dsn(), cfg["index"]["table"], cfg["embedding"]["dim"])
    check_index(store, cfg)
    embedder = Embedder(cfg["embedding"]["model"])
    judge = None if args.no_answers else Claude(cfg)
    llm = None if args.no_answers else (judge if args.answerer == "anthropic" else make_llm(cfg, args.answerer))
    server_version = llm.server_version() if args.answerer == "local" and llm is not None else None
    _, a_hash, a_path = prompts.load(cfg, "rag_answer")
    _, j_hash, j_path = prompts.load(cfg, "judge")
    quality_on = not args.no_answers and not args.no_quality
    q_prompts = {name: prompts.load(cfg, name) for name in quality.PROMPT_NAMES} if quality_on else {}
    set_path = config.path_of(cfg["eval"]["set_path"])
    documents, chunks = store.count()
    runs_dir = config.path_of("data/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    sparse_cache: dict = {}
    for method, k in itertools.product(args.method, args.top_k):
        if method not in sparse_cache:
            sparse_cache[method] = sparse_index(store, cfg, method)
        res = evaluate(cfg, questions, store, embedder, llm, k, answers=not args.no_answers,
                       with_quality=quality_on, method=method, sparse=sparse_cache[method], judge=judge)
        tag = ("" if args.answerer == "anthropic" else f"{args.answerer}_") + ("" if method == "dense" else f"{method}_")
        out = runs_dir / f"eval_{tag}top{k}.jsonl"
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in res["rows"]), encoding="utf-8")
        params = {
            "top_k": k,
            "retrieval_method": method,
            "embedding_model": cfg["embedding"]["model"],
            "chunk_size_words": cfg["chunking"]["size_words"],
            "overlap_words": cfg["chunking"]["overlap_words"],
            "index_config_hash": config.index_hash(cfg),
            "index_documents": documents,
            "index_chunks": chunks,
            "n_questions": len(questions),
            "eval_set_hash": config.file_hash(set_path),
            "llm_model": model_name(cfg, args.answerer),
            "answerer": args.answerer,
            "judge_model": cfg["llm"]["model"],
            "temperature": "model default" if args.answerer == "anthropic" else cfg["llm_local"]["temperature"],
            "max_tokens": cfg["llm"]["max_tokens"] if args.answerer == "anthropic" else cfg["llm_local"]["max_tokens"],
            "answer_prompt_hash": a_hash,
            "judge_prompt_hash": j_hash,
            "answers": not args.no_answers,
        }
        params.update({f"{name}_prompt_hash": h for name, (_, h, _) in q_prompts.items()})
        if args.answerer == "local":
            params.update({"server_version": server_version, "gpu_usd_per_hour": cfg["llm_local"]["gpu_usd_per_hour"]})
        if method in ("bm25", "hybrid"):
            params.update({"bm25_k1": cfg["bm25"]["k1"], "bm25_b": cfg["bm25"]["b"]})
        if method == "hybrid":
            params.update({"hybrid_pool": cfg["hybrid"]["pool"], "hybrid_rrf_k": cfg["hybrid"]["rrf_k"]})
        run_id = track.log_run(
            cfg["mlflow"]["rag_experiment"],
            f"rag-{tag.replace('_', '-')}top{k}" + ("-retrieval-only" if args.no_answers else ""),
            params,
            res["metrics"],
            files=[a_path, j_path, set_path, out, *[p for _, _, p in q_prompts.values()]],
            tags={"lab": "3", "kind": "rag-eval", "method": method, "answerer": args.answerer},
        )
        m = res["metrics"]
        print(f"{args.answerer} {method} top-k {k}: hit rate {m['hit_rate']:.3f}, MRR {m['mrr']:.3f}"
              + ("" if args.no_answers else f", answer score {m['answer_score']:.3f}, "
                 f"not finished {int(m['answers_not_finished'])}, cost ${m['cost_usd']:.4f}, p50 {m['answer_latency_p50_s']:.2f}s, "
                 f"p95 {m['answer_latency_p95_s']:.2f}s, max {m['answer_latency_max_s']:.2f}s, first call {m['answer_first_call_s']:.2f}s") + f"  (run {run_id[:8]})")
        if quality_on:
            print(f"        groundedness {m['groundedness']:.3f}, hallucination rate {m['hallucination_rate']:.3f}, "
                  f"declined {m['abstain_rate']:.3f}, relevance {m['answer_relevance']:.3f}, "
                  f"completeness {m['context_completeness']:.3f}, utilization {m['context_utilization']:.3f}, "
                  f"relevance on answered {m['answer_relevance_answered']:.3f}, judge cost ${m['quality_cost_usd']:.4f}")
            print(f"        parse failures: groundedness {int(m['groundedness_parse_failures'])}, "
                  f"relevance {int(m['relevance_parse_failures'])}, completeness {int(m['completeness_parse_failures'])}, "
                  f"judges not finished {int(m['judges_not_finished'])}")
            print(f"        declined although the gold post was retrieved {m['false_decline_rate']:.3f}, "
                  f"declined when it was not {m['decline_rate_when_gold_missing']:.3f}")


if __name__ == "__main__":
    main()
