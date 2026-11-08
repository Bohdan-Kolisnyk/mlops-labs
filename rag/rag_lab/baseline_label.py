"""Zero-shot LLM baseline for the Lab 2 label, logged next to the Lab 2 runs.

    uv run --project rag python -m rag_lab.baseline_label

The LLM gets a post and its time and returns the probability of an abnormal move. The metric is
the Lab 2 metric (PR-AUC) on a random sample of the test posts; the Lab 2 candidate and the
constant prior are scored on the same sample, so the three sit side by side. The run goes into
the Lab 2 experiment with the same metric name (test_pr_auc).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from rag_lab import config, prompts, track
from rag_lab.llm import Claude, ask_many, usage

PRIOR = 0.14  # fallback probability when a reply cannot be parsed


def probability_of(text: str) -> float | None:
    obj = prompts.parse_json(text) or {}
    p = obj.get("probability")
    return float(p) if isinstance(p, (int, float)) and 0 <= p <= 1 else None


def main() -> None:
    config.load_env()
    cfg = config.load()
    b = cfg["baseline"]
    test = pd.read_parquet(config.path_of(b["test_path"]))
    sample = test.sample(min(b["sample_size"], len(test)), random_state=b["seed"]).reset_index(drop=True)
    template, phash, ppath = prompts.load(cfg, "zero_shot")
    et = pd.to_datetime(sample["created_at"], utc=True).dt.tz_convert("America/New_York")
    texts = [
        prompts.render(template, time_et=t.strftime("%Y-%m-%d %H:%M"), text=x)
        for t, x in zip(et, sample["text"], strict=True)
    ]
    replies = ask_many(Claude(cfg), texts, cfg["llm"]["concurrency"])
    probs = [probability_of(r.text) for r in replies]
    failures = sum(p is None for p in probs)
    score = np.array([PRIOR if p is None else p for p in probs])
    y = sample["y"].to_numpy()

    llm_ap = float(average_precision_score(y, score))
    cand_ap = float(average_precision_score(y, sample["candidate_score"]))
    metrics = {
        "test_pr_auc": llm_ap,
        "candidate_test_pr_auc_same_sample": cand_ap,
        "prior_pr_auc_same_sample": float(y.mean()),
        "parse_failures": failures,
        **{k: v for k, v in usage(replies).items() if k != "calls"},
        "cost_per_post_usd": usage(replies)["cost_usd"] / len(replies),
    }
    run_id = track.log_run(
        cfg["mlflow"]["lab2_experiment"],
        "llm-zero-shot",
        {
            "model": cfg["llm"]["model"],
            "temperature": "model default",
            "max_tokens": cfg["llm"]["max_tokens"],
            "prompt_hash": phash,
            "sample_size": len(sample),
            "sample_seed": b["seed"],
            "positives_in_sample": int(y.sum()),
            "task": "same label as the Lab 2 model, zero-shot",
        },
        metrics,
        files=[ppath],
        tags={"lab": "3", "kind": "llm-baseline"},
    )
    print(f"{len(sample)} test posts ({int(y.sum())} positives), prompt {phash}")
    print(f"PR-AUC  LLM zero-shot {llm_ap:.3f} | Lab 2 candidate {cand_ap:.3f} | prior {y.mean():.3f}")
    print(f"parse failures {failures}, cost ${metrics['cost_usd']:.4f}, "
          f"p50 {metrics['latency_p50_s']:.2f}s, p95 {metrics['latency_p95_s']:.2f}s  (run {run_id[:8]})")


if __name__ == "__main__":
    main()
