"""Print the README tables from the MLflow runs (the latest run of each method and top-k).

    uv run --project rag python -m rag_lab.report
"""

from __future__ import annotations

import math

import mlflow

from rag_lab import config, track

ORDER = {"dense": 0, "bm25": 1, "hybrid": 2}


def latest_runs(cfg: dict):
    runs = mlflow.search_runs(
        experiment_names=[cfg["mlflow"]["rag_experiment"]],
        filter_string="tags.kind = 'rag-eval' and params.answers = 'True'",
        order_by=["attributes.start_time DESC"],
    )
    for column, name in (("params.retrieval_method", "method"), ("params.answerer", "answerer")):
        default = "dense" if name == "method" else "anthropic"
        runs[name] = runs[column].fillna(default) if column in runs else default
    runs["k"] = runs["params.top_k"].astype(int)
    runs = runs.drop_duplicates(["answerer", "method", "k"])
    runs["rank"] = runs["method"].map(ORDER)
    return runs.sort_values(["answerer", "rank", "k"])


def answer_cost_per_question(run, cfg: dict) -> float:
    """Cost of writing the answers only: the logged shadow cost for a local model, otherwise the
    logged tokens at the configured prices."""
    n = float(run["params.n_questions"])
    logged = run.get("metrics.answer_cost_usd")
    if logged is not None and not math.isnan(float(logged)):
        return float(logged) / n
    llm = cfg["llm"]
    tokens = run["metrics.answer_tokens_in"] * llm["price_per_mtok_in"] + run["metrics.answer_tokens_out"] * llm["price_per_mtok_out"]
    return float(tokens) / 1e6 / n


def main() -> None:
    cfg = config.load()
    track.setup(cfg)
    runs = latest_runs(cfg)
    print("| answerer | method | top-k | hit rate | MRR | answer score | answer cost per question (USD) | "
          "cost with judge (USD) | p50 s | p95 s | first call s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in runs.iterrows():
        print(
            f"| {r['answerer']} | {r['method']} | {r['k']} | {r['metrics.hit_rate']:.3f} | {r['metrics.mrr']:.3f} | "
            f"{r['metrics.answer_score']:.3f} | {answer_cost_per_question(r, cfg):.5f} | "
            f"{r['metrics.cost_per_question_usd']:.5f} | "
            f"{r['metrics.answer_latency_p50_s']:.2f} | {r['metrics.answer_latency_p95_s']:.2f} | "
            f"{r.get('metrics.answer_first_call_s', float('nan')):.2f} |"
        )
    if "metrics.groundedness" not in runs:
        return
    print()
    print("| answerer | method | top-k | declined | groundedness | hallucination rate | relevance on answered | "
          "completeness | utilization | declined, gold retrieved | declined, gold missing |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in runs.iterrows():
        print(
            f"| {r['answerer']} | {r['method']} | {r['k']} | {r['metrics.abstain_rate']:.3f} | {r['metrics.groundedness']:.3f} | "
            f"{r['metrics.hallucination_rate']:.3f} | {r['metrics.answer_relevance_answered']:.3f} | "
            f"{r['metrics.context_completeness']:.3f} | {r['metrics.context_utilization']:.3f} | "
            f"{r['metrics.false_decline_rate']:.3f} | {r['metrics.decline_rate_when_gold_missing']:.3f} |"
        )


if __name__ == "__main__":
    main()
