"""The LLM service (Lab 4): POST /ask over the Lab 3 retrieval pipeline, with a health contract.

    uvicorn rag_lab.service:app --host 0.0.0.0 --port 8000

A thin container: the embedding model is in the image, the index is the pgvector table of the
cluster's PostgreSQL, the answering model is an API reached over HTTP (never in the cluster).
Model, prompt files, retrieval method and top-k come from rag/config.yaml, the file of Lab 3.

/health is ready only when the API key is present, the index exists and was built with the settings
of config.yaml, the embedder runs and a probe query returns a passage. It does not call the paid
API: a health check that costs money, or that fails when the provider is slow, would take the
service out of rotation for a reason the cluster cannot fix.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from rag_lab import config, prompts
from rag_lab.embed import Embedder
from rag_lab.llm import Claude
from rag_lab.retrieval import answer_prompts, check_index, retrieve, sparse_index
from rag_lab.store import PgStore

log = logging.getLogger("llm_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RETRY_SECONDS = 5.0
BUCKETS = (0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 30)

REQUESTS = Counter("llm_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram("llm_request_latency_seconds", "Request latency", ["endpoint"], buckets=BUCKETS)
RETRIEVAL = Histogram("llm_retrieval_seconds", "Retrieval time", buckets=BUCKETS)
GENERATION = Histogram("llm_generation_seconds", "Answer generation time", buckets=BUCKETS)
TOKENS = Counter("llm_tokens_total", "Tokens of the answer calls", ["direction"])
COST = Counter("llm_cost_usd_total", "Cost of the answer calls, USD")
UPSTREAM_ERRORS = Counter("llm_upstream_errors_total", "Failed calls to the answering model")
READY = Gauge("llm_ready", "1 when the service can answer")
LAST_REQUEST = Gauge("llm_last_request_timestamp_seconds", "Time of the last /ask")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    method: str | None = Field(default=None, pattern="^(dense|bm25|hybrid)$")


class State:
    def __init__(self):
        self.cfg: dict | None = None
        self.embedder = None
        self.store = None
        self.sparse = {}
        self.llm = None
        self.template = ""
        self.prompt_hash = ""
        self.error = "starting"
        self.attempts = 0
        self.documents = 0
        self.chunks = 0
        self.ready = False


state = State()


def prepare(embedder: Embedder | None = None) -> None:
    """Everything /ask needs; raises with the reason when something is missing."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    cfg = config.load()
    store = PgStore(config.warehouse_dsn(), cfg["index"]["table"], cfg["embedding"]["dim"])
    check_index(store, cfg)
    documents, chunks = store.count()
    embedder = embedder or Embedder(cfg["embedding"]["model"])
    method = cfg["retrieval"]["method"]
    bm25 = sparse_index(store, cfg, "bm25")
    sparse = {"dense": None, "bm25": bm25, "hybrid": bm25}
    probe = retrieve(store, embedder, ["tariffs on China"], 1, method, sparse[method], cfg["hybrid"])
    if not probe[0]:
        raise RuntimeError("the probe query returned no passage: the index is empty")
    template, phash, _ = prompts.load(cfg, "rag_answer")
    llm = Claude(cfg)
    state.cfg, state.store, state.sparse, state.llm = cfg, store, sparse, llm
    state.embedder, state.template, state.prompt_hash = embedder, template, phash
    state.documents, state.chunks, state.error, state.ready = documents, chunks, "", True
    READY.set(1)
    log.info("ready: %d posts, %d chunks, model %s", documents, chunks, cfg["llm"]["model"])


def load_until_ready(stop: threading.Event) -> None:
    while not stop.is_set() and not state.ready:
        state.attempts += 1
        try:
            prepare(state.embedder)
            return
        except Exception as exc:  # noqa: BLE001
            state.error = f"{type(exc).__name__}: {exc}"[:500]
            log.warning("not ready (attempt %d): %s", state.attempts, state.error)
            if state.embedder is None:
                try:
                    state.embedder = Embedder(config.load()["embedding"]["model"])
                except Exception as load_exc:  # noqa: BLE001
                    state.error = f"embedder: {load_exc}"[:500]
            stop.wait(RETRY_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    READY.set(0)
    stop = threading.Event()
    threading.Thread(target=load_until_ready, args=(stop,), daemon=True).start()
    yield
    stop.set()


app = FastAPI(title="truth-social-rag", lifespan=lifespan)


@app.middleware("http")
async def measure(request: Request, call_next):
    path = request.url.path
    if path == "/metrics":
        return await call_next(request)
    t0, status = time.perf_counter(), "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        REQUESTS.labels(path, status).inc()
        LATENCY.labels(path).observe(time.perf_counter() - t0)


@app.get("/livez")
def livez() -> dict:
    return {"status": "alive"}


@app.get("/health")
def health() -> JSONResponse:
    if state.ready:
        return JSONResponse(
            {
                "status": "ready",
                "model": state.cfg["llm"]["model"],
                "documents": state.documents,
                "chunks": state.chunks,
            }
        )
    return JSONResponse(
        {"status": "not_ready", "reason": state.error, "attempts": state.attempts}, status_code=503
    )


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if not state.ready:
        raise HTTPException(503, f"not ready: {state.error}")
    cfg = state.cfg
    method = req.method or cfg["retrieval"]["method"]
    top_k = req.top_k or cfg["retrieval"]["top_k"]
    t0 = time.perf_counter()
    hits = retrieve(
        state.store, state.embedder, [req.question], top_k, method, state.sparse[method], cfg["hybrid"]
    )
    retrieval_s = time.perf_counter() - t0
    RETRIEVAL.observe(retrieval_s)
    prompt = answer_prompts(state.template, [req.question], hits)[0]
    try:
        reply = state.llm.ask(prompt)
    except Exception as exc:
        UPSTREAM_ERRORS.inc()
        log.exception("answering model call failed")
        raise HTTPException(502, f"the answering model failed: {type(exc).__name__}") from exc
    GENERATION.observe(reply.seconds)
    TOKENS.labels("in").inc(reply.tokens_in)
    TOKENS.labels("out").inc(reply.tokens_out)
    COST.inc(reply.cost)
    LAST_REQUEST.set(time.time())
    log.info(
        "ask method=%s k=%d retrieval=%.2fs generation=%.2fs tokens=%d/%d cost=$%.5f",
        method, top_k, retrieval_s, reply.seconds, reply.tokens_in, reply.tokens_out, reply.cost,
    )
    return {
        "answer": reply.text,
        "model": cfg["llm"]["model"],
        "method": method,
        "top_k": top_k,
        "prompt_hash": state.prompt_hash,
        "tokens_in": reply.tokens_in,
        "tokens_out": reply.tokens_out,
        "cost_usd": reply.cost,
        "retrieval_s": round(retrieval_s, 4),
        "generation_s": round(reply.seconds, 4),
        "sources": [
            {
                "post_id": h["post_id"],
                "created_at": str(h["created_at"]),
                "score": float(h["score"]),
                "text": h["text"][:300],
            }
            for h in hits[0]
        ],
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
