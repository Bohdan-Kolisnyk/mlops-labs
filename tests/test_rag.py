"""Lab 3: chunking, retrieval metrics, the index/config check and the evaluation loop.

The embedding model, pgvector and the LLM are replaced by small fakes, so these tests need
neither the network nor a key. They check our code, not the models.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))

from rag_lab import config, core, eval_rag, llm, prompts, retrieval
from rag_lab.baseline_label import probability_of
from rag_lab.store import MemoryStore

VOCAB = ["steel", "tariff", "china", "farmers", "election", "border", "oil", "fed", "bitcoin", "tax"]


class BagEmbedder:
    """Normalised word counts over a tiny vocabulary: similar words, similar vectors."""

    def encode(self, texts):
        v = np.array(
            [[t.lower().count(w) for w in VOCAB] for t in texts], dtype="float32"
        ) + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def ask(self, prompt, max_tokens=None):
        self.prompts.append(prompt)
        if "Score from 1 to 5" in prompt:
            return llm.Reply('{"score": 4, "reason": "ok"}', 100, 10, 0.2, 0.00001)
        return llm.Reply("The passage from 2025-03-12 says so.", 300, 20, 0.5, 0.00004)


def corpus():
    topics = ["steel tariff", "china tariff", "farmers tax", "election border", "oil fed", "bitcoin"]
    rows = []
    for i in range(30):
        rows.append(
            {"post_id": f"p{i:02d}", "created_at": pd.Timestamp("2025-03-01", tz="UTC") + pd.Timedelta(days=i),
             "text": f"{topics[i % 6]} update number {i} " + "words " * 5, "y": i % 2, "ret": 0.001 * i}
        )
    return pd.DataFrame(rows)


def built_store(cfg):
    ch = cfg["chunking"]
    chunks = core.build_chunks(corpus(), ch["size_words"], ch["overlap_words"], cfg["corpus"]["min_chars"])
    store = MemoryStore()
    store.rebuild(chunks, BagEmbedder().encode(chunks["text"].tolist()), config.index_hash(cfg))
    return store


def test_chunks_overlap_and_cover_the_text():
    text = " ".join(f"w{i}" for i in range(50))
    chunks = core.chunk_words(text, size=20, overlap=5)
    assert [len(c.split()) for c in chunks] == [20, 20, 20]
    assert chunks[0].split()[-5:] == chunks[1].split()[:5]
    assert set(" ".join(chunks).split()) == set(text.split())
    assert core.chunk_words("short post", 20, 5) == ["short post"]
    with pytest.raises(ValueError):
        core.chunk_words(text, 5, 5)


def test_hit_rate_and_mrr_on_a_known_ranking():
    ranked = [["a", "b", "c"], ["x", "a", "y"], ["x", "y", "z"], ["q", "r", "a"]]
    rel = [{"a"}, {"a"}, {"a"}, {"a"}]
    assert core.hit_rate(ranked, rel, 1) == 0.25
    assert core.hit_rate(ranked, rel, 3) == 0.75
    assert core.mrr(ranked, rel, 3) == pytest.approx((1 + 1 / 2 + 0 + 1 / 3) / 4)


def test_a_question_that_quotes_its_post_is_caught():
    post = "Congressman Carlos Gimenez is doing a fantastic job representing Florida"
    assert core.quotes_the_post("Who is doing a fantastic job representing Florida?", post)
    assert not core.quotes_the_post("Which Miami-area lawmaker did he praise for his work?", post)


def test_the_right_post_is_retrieved_and_ids_are_unique():
    cfg = config.load()
    store = built_store(cfg)
    hits = retrieval.retrieve(store, BagEmbedder(), ["what about the steel tariff"], 5)[0]
    assert hits[0]["post_id"] in {"p00", "p06", "p12", "p18", "p24"}
    assert len(core.unique_posts(hits)) == len(set(core.unique_posts(hits)))


def test_an_index_built_with_other_settings_is_refused():
    cfg = config.load()
    store = built_store(cfg)
    retrieval.check_index(store, cfg)
    changed = config.load()
    changed["chunking"]["size_words"] = 60
    assert config.index_hash(changed) != config.index_hash(cfg)
    with pytest.raises(RuntimeError, match="rebuild"):
        retrieval.check_index(store, changed)


def test_a_prompt_edit_changes_its_hash(tmp_path):
    cfg = config.load()
    _, h1, path = prompts.load(cfg, "judge")
    copy = tmp_path / "judge.md"
    copy.write_text(path.read_text(encoding="utf-8") + "\nBe strict.", encoding="utf-8")
    assert config.file_hash(copy) != h1


def test_the_evaluation_loop_with_fakes_gives_metrics_cost_and_latency():
    cfg = config.load()
    store = built_store(cfg)
    questions = [
        {"qid": "q1", "question": "steel tariff news", "relevant_post_ids": ["p00"], "reference_answer": "a"},
        {"qid": "q2", "question": "bitcoin news", "relevant_post_ids": ["p05"], "reference_answer": "b"},
    ]
    res = eval_rag.evaluate(cfg, questions, store, BagEmbedder(), FakeLLM(), k=10)
    m = res["metrics"]
    assert 0 <= m["hit_rate"] <= 1 and 0 <= m["mrr"] <= 1
    assert m["answer_score"] == pytest.approx(0.8)
    assert m["answer_calls"] == 2 and m["answer_tokens_in"] == 600
    assert m["cost_usd"] == pytest.approx(2 * 0.00004 + 2 * 0.00001)
    assert m["answer_latency_p50_s"] == pytest.approx(0.5)
    retrieval_only = eval_rag.evaluate(cfg, questions, store, BagEmbedder(), None, k=3, answers=False)
    assert set(retrieval_only["metrics"]) == {"hit_rate", "mrr"}


def test_the_retrieved_passages_and_their_reaction_reach_the_prompt():
    cfg = config.load()
    store = built_store(cfg)
    fake = FakeLLM()
    eval_rag.evaluate(
        cfg,
        [{"qid": "q1", "question": "oil fed", "relevant_post_ids": ["p04"], "reference_answer": "a"}],
        store, BagEmbedder(), fake, k=2,
    )
    answer_prompt = fake.prompts[0]
    assert "oil fed update" in answer_prompt and "abnormal move:" in answer_prompt


def test_the_baseline_reads_a_probability_and_rejects_nonsense():
    assert probability_of('Sure: {"probability": 0.23}') == 0.23
    assert probability_of('{"probability": 7}') is None
    assert probability_of("no idea") is None
