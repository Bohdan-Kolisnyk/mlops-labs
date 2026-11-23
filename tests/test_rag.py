"""Lab 3: chunking, retrieval metrics, the index/config check and the evaluation loop.

The embedding model, pgvector and the LLM are replaced by small fakes, so these tests need
neither the network nor a key. They check our code, not the models.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))

from rag_lab import config, core, eval_rag, llm, prompts, retrieval, sparse
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
        if "factual claims" in prompt:
            body = '{"claims": [{"claim": "a", "supported": true, "passage": 1}, {"claim": "b", "supported": false, "passage": null}]}'
            return llm.Reply(body, 500, 40, 0.3, 0.00002)
        if "Rate how much of the reference answer" in prompt:
            return llm.Reply('{"score": 1, "reason": "part"}', 400, 10, 0.2, 0.00001)
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


def test_a_changed_corpus_changes_the_index_hash(tmp_path):
    cfg = config.load()
    df = corpus()
    path = tmp_path / "corpus.parquet"
    df.to_parquet(path, index=False)
    cfg["corpus"]["path"] = str(path)
    before = config.index_hash(cfg)
    assert config.index_hash(cfg) == before
    df.loc[0, "text"] = df.loc[0, "text"] + " corrected"
    df.to_parquet(path, index=False)
    assert config.index_hash(cfg) != before


def test_claim_stats_counts_supported_claims_and_used_passages():
    claims = [
        {"claim": "a", "supported": True, "passage": 2},
        {"claim": "b", "supported": True, "passage": 2},
        {"claim": "c", "supported": False, "passage": None},
        {"claim": "d", "supported": True, "passage": 99},
    ]
    s = core.claim_stats(claims, n_passages=5)
    assert s["claims"] == 4 and s["supported"] == 3
    assert s["groundedness"] == pytest.approx(0.75)
    assert s["hallucinated"] is True
    assert s["utilization"] == pytest.approx(1 / 5)
    clean = core.claim_stats([{"claim": "a", "supported": True, "passage": 1}], n_passages=2)
    assert clean["hallucinated"] is False and clean["utilization"] == pytest.approx(0.5)


def test_an_answer_without_claims_is_an_abstention_not_a_hallucination():
    s = core.claim_stats([], n_passages=3)
    assert s["groundedness"] is None and s["hallucinated"] is None
    summary = core.quality_summary([s, core.claim_stats([{"claim": "a", "supported": True, "passage": 1}], 3)])
    assert summary["abstain_rate"] == pytest.approx(0.5)
    assert summary["groundedness"] == pytest.approx(1.0) and summary["hallucination_rate"] == pytest.approx(0.0)


def test_the_quality_judges_run_per_answer_and_are_priced_apart():
    cfg = config.load()
    store = built_store(cfg)
    questions = [
        {"qid": "q1", "question": "steel tariff news", "relevant_post_ids": ["p00"], "reference_answer": "a"},
        {"qid": "q2", "question": "bitcoin news", "relevant_post_ids": ["p05"], "reference_answer": "b"},
    ]
    fake = FakeLLM()
    res = eval_rag.evaluate(cfg, questions, store, BagEmbedder(), fake, k=5)
    m = res["metrics"]
    assert m["groundedness"] == pytest.approx(0.5)
    assert m["hallucination_rate"] == pytest.approx(1.0)
    assert m["abstain_rate"] == pytest.approx(0.0)
    assert m["answer_relevance"] == pytest.approx(0.8)
    assert m["context_completeness"] == pytest.approx(0.5)
    assert m["quality_calls"] == 6 and m["quality_parse_failures"] == 0
    assert m["quality_cost_usd"] == pytest.approx(2 * (0.00002 + 0.00001 + 0.00001))
    assert m["cost_usd"] == pytest.approx(2 * 0.00004 + 2 * 0.00001)
    assert "[1] " in fake.prompts[-1] or "[1] " in "".join(fake.prompts)
    skipped = eval_rag.evaluate(cfg, questions, store, BagEmbedder(), FakeLLM(), k=5, with_quality=False)
    assert "groundedness" not in skipped["metrics"]


def test_unparseable_judge_replies_are_counted_not_averaged():
    class Broken(FakeLLM):
        def ask(self, prompt, max_tokens=None):
            if "factual claims" in prompt or "Rate how much of the reference answer" in prompt:
                return llm.Reply("not json", 10, 2, 0.1, 0.0)
            return super().ask(prompt, max_tokens)

    cfg = config.load()
    store = built_store(cfg)
    questions = [{"qid": "q1", "question": "steel tariff news", "relevant_post_ids": ["p00"], "reference_answer": "a"}]
    m = eval_rag.evaluate(cfg, questions, store, BagEmbedder(), Broken(), k=3)["metrics"]
    assert m["quality_parse_failures"] == 2
    assert np.isnan(m["groundedness"]) and np.isnan(m["context_completeness"])
    assert m["answer_relevance"] == pytest.approx(0.8)


def test_declined_answers_are_split_by_whether_the_gold_post_was_retrieved():
    class Decliner(FakeLLM):
        def ask(self, prompt, max_tokens=None):
            if "factual claims" in prompt:
                body = '{"declines": true, "claims": [{"claim": "closest is 2024", "supported": true, "passage": 1}]}'
                return llm.Reply(body, 500, 40, 0.3, 0.00002)
            return super().ask(prompt, max_tokens)

    cfg = config.load()
    store = built_store(cfg)
    questions = [
        {"qid": "q1", "question": "steel tariff news", "relevant_post_ids": ["p00"], "reference_answer": "a"},
        {"qid": "q2", "question": "bitcoin news", "relevant_post_ids": ["p29"], "reference_answer": "b"},
    ]
    res = eval_rag.evaluate(cfg, questions, store, BagEmbedder(), Decliner(), k=1)
    m = res["metrics"]
    assert m["abstain_rate"] == pytest.approx(1.0)
    assert np.isnan(m["groundedness"]) and np.isnan(m["hallucination_rate"])
    assert [r["hit"] for r in res["rows"]] == [True, False]
    assert m["false_decline_rate"] == pytest.approx(1.0)
    assert m["decline_rate_when_gold_missing"] == pytest.approx(1.0)


def chunk_dicts(rows):
    return [
        {"chunk_id": f"{pid}#0", "post_id": pid, "chunk_no": 0, "text": text, "created_at": None, "y": 0, "ret": 0.0}
        for pid, text in rows
    ]


def test_bm25_ranks_the_keyword_match_first_and_ignores_stopwords():
    chunks = chunk_dicts([
        ("a", "[2025-03-01] steel tariff on imports from china"),
        ("b", "[2025-03-02] the farmers and the tax plan"),
        ("c", "[2025-03-03] what the oil market did"),
    ])
    bm = sparse.BM25(chunks)
    assert [h["post_id"] for h in bm.search("steel tariffs", 3)] == ["a"]
    assert bm.search("the of and", 3) == []
    assert bm.search("tax", 1)[0]["post_id"] == "b"


def test_the_date_prefix_gives_a_month_name_token():
    chunks = chunk_dicts([
        ("a", "[2025-12-05] a repost about water and sewage"),
        ("b", "[2025-03-05] a repost about water and sewage"),
    ])
    bm = sparse.BM25(chunks)
    assert bm.search("in December 2025 a repost about water", 1)[0]["post_id"] == "a"
    assert bm.search("in March 2025 a repost about water", 1)[0]["post_id"] == "b"
    assert "december" in sparse.tokens("[2025-12-05] x") and "december" not in sparse.tokens("december x")[1:]


def test_rank_fusion_favours_chunks_that_both_rankings_like():
    d = chunk_dicts([("a", "x"), ("b", "x"), ("c", "x")])
    s = chunk_dicts([("c", "x"), ("b", "x"), ("d", "x")])
    fused = sparse.rrf([d, s], k=4)
    assert {h["post_id"] for h in fused[:2]} == {"b", "c"}
    assert {h["post_id"] for h in fused[2:]} == {"a", "d"}
    assert len(sparse.rrf([d, s], k=2)) == 2
    assert all(h["score"] > 0 for h in fused)


def test_every_retrieval_method_returns_k_unique_chunks_with_the_same_fields():
    cfg = config.load()
    store = built_store(cfg)
    bm = sparse.BM25(store.all_chunks(), cfg["bm25"]["k1"], cfg["bm25"]["b"])
    for method in ("dense", "bm25", "hybrid"):
        hits = retrieval.retrieve(store, BagEmbedder(), ["steel tariff news"], 5, method, bm, cfg["hybrid"])[0]
        assert 0 < len(hits) <= 5
        assert len({h["chunk_id"] for h in hits}) == len(hits)
        assert {"post_id", "text", "y", "ret", "score"} <= set(hits[0])
    with pytest.raises(ValueError):
        retrieval.retrieve(store, BagEmbedder(), ["x"], 1, "other")


def test_the_evaluation_loop_runs_with_bm25_and_hybrid():
    cfg = config.load()
    store = built_store(cfg)
    bm = sparse.BM25(store.all_chunks(), cfg["bm25"]["k1"], cfg["bm25"]["b"])
    questions = [{"qid": "q1", "question": "steel tariff news", "relevant_post_ids": ["p00"], "reference_answer": "a"}]
    for method in ("bm25", "hybrid"):
        res = eval_rag.evaluate(cfg, questions, store, BagEmbedder(), FakeLLM(), k=5, method=method, sparse=bm)
        assert res["metrics"]["hit_rate"] == 1.0
        assert "groundedness" in res["metrics"]


class _FakeVllm(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps(
            {"choices": [{"message": {"content": "hello"}, "finish_reason": "length"}],
             "usage": {"prompt_tokens": 12, "completion_tokens": 7}}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        body = json.dumps({"version": "0.99"}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_the_local_llm_reads_an_openai_style_reply_and_the_server_version():
    server = HTTPServer(("127.0.0.1", 0), _FakeVllm)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg = config.load()
        cfg["llm_local"]["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
        local = llm.make_llm(cfg, "local")
        reply = local.ask("hi")
        assert (reply.text, reply.tokens_in, reply.tokens_out, reply.cost) == ("hello", 12, 7, 0.0)
        assert reply.stop_reason == "max_tokens"
        assert local.server_version() == "0.99"
    finally:
        server.shutdown()


def test_an_unreachable_local_server_gives_an_actionable_error():
    cfg = config.load()
    cfg["llm_local"]["base_url"] = "http://127.0.0.1:9/v1"
    with pytest.raises(RuntimeError, match="cannot reach the local LLM server"):
        llm.make_llm(cfg, "local").ask("hi")
    with pytest.raises(ValueError):
        llm.make_llm(cfg, "other")


def test_a_local_answerer_is_priced_by_wall_time_and_the_judges_stay_with_the_api():
    class Local(FakeLLM):
        gpu_usd_per_hour = 3600.0

    cfg = config.load()
    store = built_store(cfg)
    questions = [{"qid": "q1", "question": "steel tariff news", "relevant_post_ids": ["p00"], "reference_answer": "a"}]
    judge = FakeLLM()
    m = eval_rag.evaluate(cfg, questions, store, BagEmbedder(), Local(), k=3, judge=judge)["metrics"]
    assert m["answer_cost_usd"] == pytest.approx(m["answer_wall_s"])
    assert m["cost_usd"] == pytest.approx(m["answer_wall_s"] + 0.00001)
    assert any("Score from 1 to 5" in p for p in judge.prompts) and len(judge.prompts) == 4
