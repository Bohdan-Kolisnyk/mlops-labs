"""The ML service: the champion of the registry behind /predict, with a health contract.

    uvicorn mlops_labs.serve.ml_service:app --host 0.0.0.0 --port 8000

/health   readiness: 200 only when a model is loaded AND has scored a smoke row; else 503
/livez    liveness: the process answers (a missing model is a readiness problem, not a reason to
          restart the container)
/predict  rows in, probability and alert flag out; every served row is appended to the request
          log, which is the current window of the drift job
/feedback labels that arrive later, joined to the log by request id
/metrics  Prometheus
/admin/reload  load the alias again without a restart (used by the retrain action)

The model is loaded from the registry by name and alias in a background thread that retries
until it succeeds, so a registry that is not up yet delays readiness and never crashes the pod.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from mlops_labs import models
from mlops_labs.serve import common

log = logging.getLogger("ml_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RETRY_SECONDS = 5.0

REQUESTS = Counter("ml_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram(
    "ml_request_latency_seconds",
    "Request latency",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
ROWS = Counter("ml_predicted_rows_total", "Rows scored")
ALERT_ROWS = Counter("ml_alert_rows_total", "Rows scored above the alert threshold")
PROBABILITY = Histogram(
    "ml_prediction_probability",
    "Predicted probability of an abnormal move",
    buckets=tuple(i / 20 for i in range(1, 20)),
)
LOG_ERRORS = Counter("ml_request_log_errors_total", "Served rows that could not be logged")
LABELS = Counter("ml_feedback_labels_total", "Labels received")
MODEL_LOADED = Gauge("ml_model_loaded", "1 when a model is loaded and ready")
MODEL_INFO = Gauge("ml_model_info", "The loaded model version", ["version", "alias", "git_commit"])
LAST_REQUEST = Gauge("ml_last_request_timestamp_seconds", "Time of the last served request")
LOAD_FAILURES = Counter("ml_model_load_failures_total", "Failed attempts to load the model")


class Instance(BaseModel):
    text: str
    text_len: float
    link_count: float
    caps_share: float
    has_media: float
    slot_index: float
    weekday: float
    minutes_since_open: float
    sigma: float
    last_bar_return: float | None = 0.0
    posts_prev_60min: float


class PredictRequest(BaseModel):
    instances: list[Instance] = Field(min_length=1, max_length=1000)


class Label(BaseModel):
    request_id: str
    y: int = Field(ge=0, le=1)


class FeedbackRequest(BaseModel):
    labels: list[Label] = Field(min_length=1, max_length=5000)


SMOKE_ROW = {
    "text": "Tariffs on steel and aluminum start tomorrow.",
    "text_len": 44.0,
    "link_count": 0.0,
    "caps_share": 0.05,
    "has_media": 0.0,
    "slot_index": 1.0,
    "weekday": 2.0,
    "minutes_since_open": 90.0,
    "sigma": 0.003,
    "last_bar_return": 0.0,
    "posts_prev_60min": 0.0,
}


class State:
    """What the service currently serves. Replaced as a whole, never mutated half way."""

    def __init__(self):
        self.model = None
        self.info: dict = {}
        self.error: str = "model not loaded yet"
        self.attempts = 0
        self.lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None


state = State()
request_log: common.RequestLog | None = None


def to_frame(rows: list[dict]) -> pd.DataFrame:
    return models.prepare_candidate_input(pd.DataFrame(rows))


def load_champion() -> tuple[object, dict]:
    """The alias of the registry resolved to a version, loaded, and proven by one prediction."""
    name, alias = common.model_name(), common.model_alias()
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)
    threshold = mv.tags.get("alert_threshold")
    if threshold is None:
        raise RuntimeError(f"{name} version {mv.version} has no alert_threshold tag")
    model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    smoke = float(model.predict_proba(to_frame([SMOKE_ROW]))[:, 1][0])
    if not 0.0 <= smoke <= 1.0:
        raise RuntimeError(f"smoke prediction {smoke} is not a probability")
    info = {
        "name": name,
        "alias": alias,
        "version": str(mv.version),
        "data_version": mv.tags.get("data_version", "?"),
        "git_commit": mv.tags.get("git_commit", "?"),
        "alert_threshold": float(threshold),
        "image_tag": os.environ.get("IMAGE_TAG", "dev"),
    }
    return model, info


def install(model: object, info: dict) -> None:
    with state.lock:
        old = state.info
        state.model, state.info, state.error = model, info, ""
    if old:
        MODEL_INFO.remove(old["version"], old["alias"], old["git_commit"])
    MODEL_INFO.labels(info["version"], info["alias"], info["git_commit"]).set(1)
    MODEL_LOADED.set(1)
    log.info("serving %s version %s (alias %s)", info["name"], info["version"], info["alias"])


def load_until_ready(stop: threading.Event) -> None:
    while not stop.is_set() and not state.ready:
        state.attempts += 1
        try:
            install(*load_champion())
            return
        except Exception as exc:  # noqa: BLE001
            state.error = f"{type(exc).__name__}: {exc}"[:500]
            LOAD_FAILURES.inc()
            log.warning("model not loaded (attempt %d): %s", state.attempts, state.error)
            stop.wait(RETRY_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global request_log
    request_log = common.RequestLog(common.request_dir())
    MODEL_LOADED.set(0)
    stop = threading.Event()
    thread = threading.Thread(target=load_until_ready, args=(stop,), daemon=True)
    thread.start()
    yield
    stop.set()


app = FastAPI(title="truth-social-abnormal-move", lifespan=lifespan)


@app.middleware("http")
async def measure(request: Request, call_next):
    path = request.url.path
    if path == "/metrics":
        return await call_next(request)
    t0 = time.perf_counter()
    status = "500"
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
        return JSONResponse({"status": "ready", "model_version": state.info["version"]})
    return JSONResponse(
        {"status": "not_ready", "reason": state.error, "attempts": state.attempts},
        status_code=503,
    )


@app.get("/model")
def model_info() -> dict:
    if not state.ready:
        raise HTTPException(503, state.error)
    return state.info


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    with state.lock:
        model, info = state.model, dict(state.info)
    if model is None:
        raise HTTPException(503, f"model not loaded: {state.error}")
    rows = [i.model_dump() for i in req.instances]
    proba = model.predict_proba(to_frame(rows))[:, 1]
    now = common.utc_now()
    threshold = info["alert_threshold"]
    ids = [uuid.uuid4().hex for _ in rows]
    out, records = [], []
    for rid, row, p in zip(ids, rows, proba, strict=True):
        alert = bool(p >= threshold)
        out.append({"request_id": rid, "probability": float(p), "alert": alert})
        records.append(
            {
                "ts": now.isoformat(),
                "request_id": rid,
                "model_version": info["version"],
                **row,
                "probability": float(p),
                "alert": alert,
            }
        )
        PROBABILITY.observe(float(p))
    try:
        request_log.append_requests(records)
    except OSError:
        LOG_ERRORS.inc(len(records))
        log.exception("could not write the request log")
    ROWS.inc(len(rows))
    ALERT_ROWS.inc(sum(o["alert"] for o in out))
    LAST_REQUEST.set(now.timestamp())
    log.info("predict rows=%d version=%s alerts=%d", len(rows), info["version"], sum(o["alert"] for o in out))
    return {
        "model_version": info["version"],
        "alias": info["alias"],
        "image_tag": info["image_tag"],
        "alert_threshold": threshold,
        "predictions": out,
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    now = common.utc_now().isoformat()
    request_log.append_labels(
        [{"ts": now, "request_id": lab.request_id, "y": lab.y} for lab in req.labels]
    )
    LABELS.inc(len(req.labels))
    return {"accepted": len(req.labels)}


@app.post("/admin/reload")
def reload_model() -> dict:
    try:
        model, info = load_champion()
    except Exception as exc:
        raise HTTPException(500, f"reload failed, still serving the old model: {exc}") from exc
    install(model, info)
    return info


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


