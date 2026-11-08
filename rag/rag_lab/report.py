"""Print the README tables from the MLflow runs.

    uv run --project rag python -m rag_lab.report
"""

from __future__ import annotations

import mlflow

from rag_lab import config, track


def main() -> None:
    cfg = config.load()
    track.setup(cfg)
    runs = mlflow.search_runs(
        experiment_names=[cfg["mlflow"]["rag_experiment"]],
        filter_string="tags.kind = 'rag-eval' and params.answers = 'True'",
        order_by=["attributes.start_time DESC"],
    )
    runs = runs.drop_duplicates("params.top_k").sort_values("params.top_k", key=lambda s: s.astype(int))
    print("| top-k | hit rate | MRR | answer score | cost per question (USD) | p50 s | p95 s |")
    print("|---|---|---|---|---|---|---|")
    for _, r in runs.iterrows():
        print(
            f"| {r['params.top_k']} | {r['metrics.hit_rate']:.3f} | {r['metrics.mrr']:.3f} | "
            f"{r['metrics.answer_score']:.3f} | {r['metrics.cost_per_question_usd']:.5f} | "
            f"{r['metrics.answer_latency_p50_s']:.2f} | {r['metrics.answer_latency_p95_s']:.2f} |"
        )


if __name__ == "__main__":
    main()
