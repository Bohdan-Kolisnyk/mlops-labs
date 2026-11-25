"""Lab 4: the service contract, the registry bootstrap, the request log, the monitor helpers and
the retrain responder. MLflow runs on a SQLite file, Kubernetes and the answering model are
fakes, so the tests need neither a cluster nor the network.

Run with the serving packages:
    uv run --with fastapi --with httpx --with prometheus-client --with alembic pytest tests/test_serve.py
"""

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
pytest.importorskip("alembic")

import mlflow
from fastapi.testclient import TestClient

from mlops_labs.features import FEATURE_COLUMNS
from mlops_labs.serve import bootstrap, common, ml_service, monitor, responder

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "deploy" / "seed"
WORDS = ["tariff", "china", "steel", "market", "fed", "deal", "great", "fake", "news", "border"]


def synthetic_table(n: int = 360, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = pd.DataFrame(
        {
            "text": [" ".join(rng.choice(WORDS, 6)) for _ in range(n)],
            "text_len": rng.integers(20, 400, n).astype(float),
            "link_count": rng.integers(0, 2, n).astype(float),
            "caps_share": rng.random(n) * 0.3,
            "has_media": rng.integers(0, 2, n).astype(float),
            "slot_index": rng.integers(0, 5, n).astype(float),
            "weekday": rng.integers(0, 5, n).astype(float),
            "minutes_since_open": rng.integers(0, 300, n).astype(float),
            "sigma": 0.003 + rng.random(n) * 0.002,
            "last_bar_return": rng.normal(0, 0.002, n),
            "posts_prev_60min": rng.integers(0, 4, n).astype(float),
            "y": rng.integers(0, 2, n),
        }
    )
    rows["split"] = ["train"] * (n * 2 // 3) + ["val"] * (n // 6) + ["test"] * (n - n * 2 // 3 - n // 6)
    return rows


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "art"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GIT_COMMIT", "abc123")
    monkeypatch.setenv("IMAGE_TAG", "abc123")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(ml_service, "state", ml_service.State())
    return tmp_path


def register(table: pd.DataFrame, alias: str | None = "champion", c: float = 1.0) -> dict:
    return bootstrap.train_and_register(
        table, "data-v1", c, 0, alias, experiment=f"test-{time.time_ns()}"
    )


def instance(**over) -> dict:
    row = dict(ml_service.SMOKE_ROW)
    row.update(over)
    return row


def wait_ready(client: TestClient, seconds: float = 30) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if client.get("/health").status_code == 200:
            return
        time.sleep(0.2)
    raise AssertionError(client.get("/health").text)


def test_not_ready_without_a_model_and_alive(registry, monkeypatch):
    monkeypatch.setattr(ml_service, "RETRY_SECONDS", 0.2)
    with TestClient(ml_service.app) as client:
        time.sleep(0.5)
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["status"] == "not_ready"
        assert health.json()["reason"]
        assert client.get("/livez").status_code == 200
        assert client.post("/predict", json={"instances": [instance()]}).status_code == 503


def test_ready_after_the_alias_appears_and_predict_is_logged(registry, monkeypatch):
    monkeypatch.setattr(ml_service, "RETRY_SECONDS", 0.2)
    table = synthetic_table()
    with TestClient(ml_service.app) as client:
        time.sleep(0.4)
        assert client.get("/health").status_code == 503
        register(table)
        wait_ready(client)
        body = client.post("/predict", json={"instances": [instance(), instance(sigma=0.01)]}).json()
        assert body["model_version"] == "1" and body["image_tag"] == "abc123"
        assert len(body["predictions"]) == 2
        assert all(0 <= p["probability"] <= 1 for p in body["predictions"])
        ids = [p["request_id"] for p in body["predictions"]]
        assert client.post("/feedback", json={"labels": [{"request_id": ids[0], "y": 1}]}).json() == {
            "accepted": 1
        }
    log = common.RequestLog(common.request_dir())
    requests = log.requests()
    assert len(requests) == 2
    assert set(requests["request_id"]) == set(ids)
    assert {"text", *FEATURE_COLUMNS, "probability", "alert", "model_version", "ts"} <= set(requests.columns)
    assert len(log.labels()) == 1


def test_alias_decides_the_version_and_reload_follows_it(registry, monkeypatch):
    monkeypatch.setattr(ml_service, "RETRY_SECONDS", 0.2)
    table = synthetic_table()
    register(table)
    with TestClient(ml_service.app) as client:
        wait_ready(client)
        assert client.get("/model").json()["version"] == "1"
        second = register(table, alias=None, c=0.1)
        assert client.get("/model").json()["version"] == "1"
        mlflow.MlflowClient().set_registered_model_alias(common.model_name(), "champion", second["version"])
        assert client.post("/admin/reload").json()["version"] == "2"
        body = client.post("/predict", json={"instances": [instance()]}).json()
        assert body["model_version"] == "2"


def test_a_row_with_a_missing_feature_is_rejected(registry, monkeypatch):
    monkeypatch.setattr(ml_service, "RETRY_SECONDS", 0.2)
    register(synthetic_table())
    with TestClient(ml_service.app) as client:
        wait_ready(client)
        bad = instance()
        del bad["sigma"]
        assert client.post("/predict", json={"instances": [bad]}).status_code == 422
        assert client.post("/predict", json={"instances": []}).status_code == 422


def test_metrics_endpoint_has_the_service_metrics(registry, monkeypatch):
    monkeypatch.setattr(ml_service, "RETRY_SECONDS", 0.2)
    register(synthetic_table())
    with TestClient(ml_service.app) as client:
        wait_ready(client)
        client.post("/predict", json={"instances": [instance()]})
        text = client.get("/metrics").text
    for name in ("ml_requests_total", "ml_request_latency_seconds_bucket", "ml_model_info",
                 "ml_predicted_rows_total", "ml_model_loaded"):
        assert name in text


def test_bootstrap_if_missing_is_idempotent_and_writes_the_reference(registry, tmp_path, monkeypatch):
    seed = tmp_path / "seed"
    seed.mkdir()
    synthetic_table().to_parquet(seed / "train_table.parquet", index=False)
    (seed / "DATA_VERSION").write_text("v1\n")
    monkeypatch.setenv("SEED_DIR", str(seed))
    assert bootstrap.main(["--if-missing"]) == 0
    versions = mlflow.MlflowClient().search_model_versions(f"name='{common.model_name()}'")
    assert len(versions) == 1
    tags = versions[0].tags
    assert tags["git_commit"] == "abc123" and float(tags["alert_threshold"]) > 0
    ref = pd.read_parquet(common.reference_dir() / "reference.parquet")
    assert len(ref) > 0 and "probability" in ref.columns
    assert bootstrap.main(["--if-missing"]) == 0
    assert len(mlflow.MlflowClient().search_model_versions(f"name='{common.model_name()}'")) == 1


def test_monitor_quality_needs_enough_labels():
    n = 60
    window = pd.DataFrame(
        {
            "request_id": [str(i) for i in range(n)],
            "probability": np.linspace(0, 1, n),
            "alert": [i > 45 for i in range(n)],
        }
    )
    labels = pd.DataFrame(
        {"request_id": window["request_id"], "y": [1 if i % 3 == 0 else 0 for i in range(n)],
         "ts": pd.Timestamp("2026-01-01", tz="UTC")}
    )
    few = monitor.quality(window, labels.head(10), {})
    assert few == {"labeled_rows": 10.0}
    full = monitor.quality(window, labels, {"val_pr_auc": 0.2, "val_base_rate": 0.1})
    assert full["labeled_rows"] == 60.0 and 0 <= full["pr_auc"] <= 1
    assert full["reference_pr_auc"] == 0.2
    assert monitor.quality(window, pd.DataFrame(), {}) == {"labeled_rows": 0.0}


def test_monitor_metric_id_parsing():
    match = monitor.COLUMN_ID.match("ValueDrift(column=sigma,method=ks,threshold=0.05)")
    assert match and match.group(1) == "sigma"
    assert monitor.COLUMN_ID.match("ValueDrift(column=text_len)").group(1) == "text_len"


def test_request_log_roundtrip(tmp_path):
    log = common.RequestLog(tmp_path)
    assert log.requests().empty and log.labels().empty
    log.append_requests([{"ts": "2026-01-01T00:00:00+00:00", "request_id": "a", "text": "x, y", "sigma": 0.1}])
    log.append_labels([{"ts": "2026-01-01T00:01:00+00:00", "request_id": "a", "y": 1}])
    df = log.requests()
    assert df.loc[0, "request_id"] == "a" and str(df.loc[0, "ts"].tzinfo) == "UTC"
    assert log.labels().loc[0, "y"] == 1


class FakeKube:
    def __init__(self, jobs=None):
        self.jobs = jobs or []
        self.created = []

    def __call__(self, method, path, body=None):
        if method == "GET":
            return {"items": self.jobs}
        self.created.append(body)
        return body


def firing(name):
    return {"alerts": [{"status": "firing", "labels": {"alertname": name},
                        "annotations": {"summary": "s"}}]}


def test_responder_starts_one_retrain_job_for_a_trigger_alert(monkeypatch):
    fake = FakeKube()
    monkeypatch.setattr(responder, "kube", fake)
    monkeypatch.setattr(responder, "namespace", lambda: "mlops")
    monkeypatch.setenv("TRIGGER_ALERTS", "FeatureDriftSigma,DriftShareHigh")
    monkeypatch.setenv("RETRAIN_IMAGE", "mlops-labs-ml:abc")
    monkeypatch.setenv("APP_CONFIGMAP", "app-config")
    assert responder.handle_alerts(firing("MLServiceDown")) == []
    started = responder.handle_alerts(firing("FeatureDriftSigma"))
    assert len(started) == 1 and fake.created[0]["metadata"]["labels"] == {"app": "retrain"}
    spec = fake.created[0]["spec"]["template"]["spec"]["containers"][0]
    assert spec["args"] == ["--trigger", "alert:FeatureDriftSigma"]
    assert spec["image"] == "mlops-labs-ml:abc"


def test_responder_respects_the_cooldown(monkeypatch):
    now = pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z")
    fake = FakeKube(jobs=[{"metadata": {"name": "retrain-old", "creationTimestamp": now}}])
    monkeypatch.setattr(responder, "kube", fake)
    monkeypatch.setattr(responder, "namespace", lambda: "mlops")
    monkeypatch.setenv("TRIGGER_ALERTS", "FeatureDriftSigma")
    monkeypatch.setenv("RETRAIN_IMAGE", "mlops-labs-ml:abc")
    monkeypatch.setenv("APP_CONFIGMAP", "app-config")
    assert responder.handle_alerts(firing("FeatureDriftSigma")) == []
    assert fake.created == []


def test_replay_shift_changes_exactly_the_requested_fraction():
    pytest.importorskip("badgers")
    sys.path.insert(0, str(ROOT / "scripts"))
    import replay

    df = synthetic_table(200)
    result = replay.shifted(df, ["sigma"], 3.0, 0.4, seed=1)
    assert result["description"]["rows_affected"] == 80
    assert (result["frame"]["sigma"] > 0).all()
    assert result["frame"]["sigma"].mean() > df["sigma"].mean()
    assert result["frame"]["text_len"].equals(df["text_len"])
