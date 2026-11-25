"""Lab 4: the LLM service contract with a fake index, embedder and answering model."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("prometheus_client")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))

from fastapi.testclient import TestClient
from rag_lab import llm, service
from rag_lab.store import MemoryStore

VOCAB = ["steel", "tariff", "china", "farmers", "election", "border", "oil", "fed", "bitcoin", "tax"]


class BagEmbedder:
    def __init__(self, *_):
        pass

    def encode(self, texts):
        v = np.array([[t.lower().count(w) for w in VOCAB] for t in texts], dtype="float32") + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class FakeClaude:
    def __init__(self, cfg):
        self.cfg = cfg

    def ask(self, prompt, max_tokens=None):
        return llm.Reply("Tariffs on steel were announced (2025-03-12).", 120, 14, 0.01, 0.0001)


def make_store() -> MemoryStore:
    texts = ["steel tariff announced today", "china tariff on farmers", "oil and the fed", "tax and the border"]
    chunks = pd.DataFrame(
        {
            "chunk_id": [f"p{i}-0" for i in range(len(texts))],
            "post_id": [f"p{i}" for i in range(len(texts))],
            "chunk_no": 0,
            "text": texts,
            "created_at": pd.Timestamp("2025-03-12", tz="UTC"),
            "y": [1, 0, 0, 1],
            "ret": [0.01, -0.002, 0.0, 0.004],
        }
    )
    store = MemoryStore()
    store.rebuild(chunks, BagEmbedder().encode(texts), "hash")
    return store


@pytest.fixture()
def fresh(monkeypatch):
    monkeypatch.setattr(service, "state", service.State())
    monkeypatch.setattr(service, "RETRY_SECONDS", 0.2)
    monkeypatch.setenv("WAREHOUSE_URL", "postgresql+psycopg2://u:p@localhost:5432/warehouse")
    monkeypatch.setattr(service, "PgStore", lambda *a, **k: make_store())
    monkeypatch.setattr(service, "check_index", lambda store, cfg: None)
    monkeypatch.setattr(service, "Embedder", BagEmbedder)
    monkeypatch.setattr(service, "Claude", FakeClaude)


def wait_ready(client: TestClient) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if client.get("/health").status_code == 200:
            return
        time.sleep(0.2)
    raise AssertionError(client.get("/health").text)


def test_not_ready_without_the_api_key(fresh, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with TestClient(service.app) as client:
        time.sleep(0.6)
        health = client.get("/health")
        assert health.status_code == 503 and "ANTHROPIC_API_KEY" in health.json()["reason"]
        assert client.get("/livez").status_code == 200
        assert client.post("/ask", json={"question": "What about steel tariffs?"}).status_code == 503


def test_ready_and_ask_returns_answer_sources_and_cost(fresh, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with TestClient(service.app) as client:
        wait_ready(client)
        health = client.get("/health").json()
        assert health["chunks"] == 4 and health["documents"] == 4
        reply = client.post("/ask", json={"question": "What about steel tariffs?", "top_k": 2}).json()
        assert reply["answer"].startswith("Tariffs on steel")
        assert len(reply["sources"]) == 2 and reply["sources"][0]["post_id"] == "p0"
        assert reply["top_k"] == 2 and reply["tokens_in"] == 120 and reply["cost_usd"] == 0.0001
        assert reply["model"] and reply["prompt_hash"]
        text = client.get("/metrics").text
        for name in ("llm_requests_total", "llm_retrieval_seconds_bucket", "llm_cost_usd_total", "llm_ready"):
            assert name in text


def test_bad_requests_are_rejected(fresh, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with TestClient(service.app) as client:
        wait_ready(client)
        assert client.post("/ask", json={"question": "x"}).status_code == 422
        assert client.post("/ask", json={"question": "What about steel?", "top_k": 99}).status_code == 422
        assert client.post("/ask", json={"question": "What about steel?", "method": "x"}).status_code == 422
