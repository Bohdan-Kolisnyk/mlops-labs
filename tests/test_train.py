"""End to end on synthetic data: sources agree, runs are tracked with lineage, the model is
registered with both tags and the alias, and an unversioned dataset cannot be registered."""

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient
from sqlalchemy import create_engine

from mlops_labs import dataset, lineage
from mlops_labs import pipeline as P
from mlops_labs import train as T
from tests.conftest import make_bars, make_posts, write_raw

DATA_VERSION = "0123456789abcdef0123456789abcdef.dir"


@pytest.fixture()
def raw(tmp_path):
    # about 15 months of trading days, so that train, validation and test all exist
    bars = make_bars(days=330, start="2025-01-02")
    posts = make_posts(bars, n=5000)
    # make_posts repeats texts on purpose; here almost every text should be usable
    keep_empty = posts["text"] == ""
    posts["text"] = [f"post {i} about tariff or weather" for i in range(len(posts))]
    posts.loc[keep_empty, "text"] = ""
    raw_dir = write_raw(tmp_path, bars, posts)
    (tmp_path / "raw.dvc").write_text(f"outs:\n- md5: {DATA_VERSION}\n  path: raw\n")
    return raw_dir


@pytest.fixture()
def tracking(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(lineage, "git_commit", lambda: "a" * 40)
    monkeypatch.setattr(lineage, "git_dirty", lambda: False)
    yield
    mlflow.set_tracking_uri("")


def test_raw_and_warehouse_sources_give_the_same_training_table(raw, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/wh.db")
    staging = str(tmp_path / "staging")
    created = pd.read_parquet(f"{raw}/train-00000-of-00001.parquet")["datetime"]
    last = pd.to_datetime(created, utc=True).max().strftime("%Y-%m")
    for month in pd.period_range("2025-01", last, freq="M").strftime("%Y-%m"):
        start, end, batch = P.batch_bounds(month)
        P.extract(raw, staging, start, end, batch)
        P.validate(raw, staging, batch)
        P.transform(raw, staging, start, end, batch)
        P.load(engine, staging, batch)
        P.build_features(engine, raw, batch)
    from_raw, v_raw = dataset.from_raw(raw)
    from_wh, v_wh = dataset.from_warehouse(engine)
    assert v_raw == v_wh == DATA_VERSION
    pd.testing.assert_frame_equal(from_raw, from_wh, check_dtype=False)


def test_runs_carry_both_lineage_tags_and_the_model_is_registered(raw, tracking):
    result = T.run_training(source="raw", raw_dir=raw, bootstrap_n=20, register=True)
    client = MlflowClient()
    assert set(result["run_ids"]) == {
        "B0-prior",
        "B1-keywords",
        "B2-context",
        "candidate-tfidf-lr",
    }
    for run_id in result["run_ids"].values():
        tags = client.get_run(run_id).data.tags
        assert tags["data_version"] == DATA_VERSION
        assert tags["git_commit"] == "a" * 40
    mv = client.get_model_version_by_alias(T.MODEL_NAME, T.ALIAS)
    assert mv.tags["data_version"] == DATA_VERSION
    assert mv.tags["git_commit"] == "a" * 40
    model = mlflow.sklearn.load_model(f"models:/{T.MODEL_NAME}@{T.ALIAS}")
    x = dataset.from_raw(raw)[0].head(3)
    from mlops_labs.models import prepare_candidate_input

    p = model.predict_proba(prepare_candidate_input(x))[:, 1]
    assert ((0 <= p) & (p <= 1)).all()


def test_same_data_and_seed_reproduce_the_metrics_exactly(raw, tracking):
    a = T.run_training(source="raw", raw_dir=raw, bootstrap_n=20)
    b = T.run_training(source="raw", raw_dir=raw, bootstrap_n=20)
    for key in T.CHECKED_METRICS:
        assert a["metrics"]["candidate-tfidf-lr"][key] == b["metrics"]["candidate-tfidf-lr"][key]


def test_unversioned_data_or_dirty_code_cannot_be_registered(raw, tracking, tmp_path, monkeypatch):
    (tmp_path / "raw.dvc").unlink()
    with pytest.raises(T.LineageError, match="data version"):
        T.run_training(source="raw", raw_dir=raw, bootstrap_n=20, register=True)
    (tmp_path / "raw.dvc").write_text(f"outs:\n- md5: {DATA_VERSION}\n  path: raw\n")
    monkeypatch.setattr(lineage, "git_dirty", lambda: True)
    with pytest.raises(T.LineageError, match="uncommitted"):
        T.run_training(source="raw", raw_dir=raw, bootstrap_n=20, register=True)


def test_a_stale_warehouse_cannot_be_registered(raw, tracking, monkeypatch):
    table, _ = dataset.from_raw(raw)
    monkeypatch.setattr("mlops_labs.warehouse.get_engine", lambda: None)
    monkeypatch.setattr(dataset, "from_warehouse", lambda engine: (table, "f" * 32 + ".dir"))
    with pytest.raises(T.LineageError, match="warehouse was built from"):
        T.run_training(source="warehouse", raw_dir=raw, bootstrap_n=20, register=True)
