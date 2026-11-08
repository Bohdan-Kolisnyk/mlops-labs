"""Write the Lab 3 inputs from the versioned raw data (derived files, git-ignored).

    uv run python scripts/export_rag_corpus.py

rag/data/corpus.parquet     the retrieval corpus: every labelled post with text, once (a repeated
                            text keeps its first post), with the measured reaction: y (abnormal
                            move) and ret (log return of the next two hourly bars);
rag/data/test_posts.parquet the test posts with text, their label and the score of the Lab 2
                            champion model, for the zero-shot LLM baseline comparison.
Needs the stack up (the champion is loaded from the MLflow registry).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops_labs import config, dataset, models, predict
from mlops_labs.data import load_bars, load_posts
from mlops_labs.env import ROOT, load_env
from mlops_labs.labels import build_windows, label_posts

OUT = ROOT / "rag" / "data"


def main() -> None:
    load_env()
    raw = ROOT / "data" / "raw"
    OUT.mkdir(parents=True, exist_ok=True)

    posts = load_posts(raw / config.POSTS_FILE)
    labelled = label_posts(posts, build_windows(load_bars(raw / config.BARS_FILE)))
    keep = labelled[~labelled["text_empty"] & ~labelled["dup_text"]]
    corpus = keep[["post_id", "created_at", "text", "y", "r"]].rename(columns={"r": "ret"})
    corpus.to_parquet(OUT / "corpus.parquet", index=False)

    table, _ = dataset.from_raw(raw)
    test = table[dataset.split_name(table["session_date"]) == "test"]
    test = test[test["post_id"].isin(corpus["post_id"])].copy()
    model, _ = predict.load_model()
    test["candidate_score"] = model.predict_proba(models.prepare_candidate_input(test))[:, 1]
    test[["post_id", "created_at", "text", "y", "candidate_score"]].to_parquet(
        OUT / "test_posts.parquet", index=False
    )
    print(f"corpus {len(corpus)} posts, test {len(test)} posts -> {OUT}")


if __name__ == "__main__":
    main()
