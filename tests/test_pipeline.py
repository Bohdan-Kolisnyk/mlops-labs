import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from mlops_labs import pipeline as P
from mlops_labs.data import SchemaError, load_posts
from mlops_labs.quality import QualityGateError
from mlops_labs.warehouse import count_batch
from tests.conftest import make_bars, make_posts, write_raw


@pytest.fixture()
def env(tmp_path):
    bars = make_bars(days=120)
    posts = make_posts(bars, n=800)
    raw = write_raw(tmp_path, bars, posts)
    engine = create_engine(f"sqlite:///{tmp_path}/wh.db")
    return raw, engine


def run_month(env, month):
    raw, engine = env
    start, end, batch = P.batch_bounds(month)
    P.extract(raw, engine, start, end, batch)
    P.validate(raw, engine, batch)
    stats = P.transform(raw, engine, start, end, batch)
    P.quality_check(raw, engine, start, end, batch)
    counts = P.load(engine, batch)
    return start, end, batch, stats, counts


def test_batch_comes_from_the_parameter_or_the_interval_never_the_clock():
    start, end, name = P.batch_bounds("2025-03")
    utc = "UTC"
    assert start == pd.Timestamp("2025-03-01", tz=utc)
    assert end == pd.Timestamp("2025-04-01", tz=utc)
    assert name == "2025-03"
    may, june = pd.Timestamp("2025-05-01", tz=utc), pd.Timestamp("2025-06-01", tz=utc)
    _, end_from_interval, name_from_interval = P.batch_bounds(None, may, june)
    assert name_from_interval == "2025-05" and end_from_interval == june
    # Airflow 3 without a timetable: start == end, the month is still one month long
    _, end_same, _ = P.batch_bounds(None, may, may)
    assert end_same == june


def test_rows_belong_to_the_interval_and_only_to_it(env):
    start, end, batch, _, counts = run_month(env, "2024-03")
    clean = P.read_clean_batch(env[1], batch)
    assert len(clean) == counts["clean_posts"] > 0
    assert clean["created_at"].between(start, end, inclusive="left").all()


def test_rerunning_the_same_interval_does_not_duplicate_rows(env):
    *_, counts_first = run_month(env, "2024-03")
    *_, counts_second = run_month(env, "2024-03")
    assert counts_first == counts_second


def test_other_batches_are_untouched_by_a_rerun(env):
    *_, c_march = run_month(env, "2024-03")
    run_month(env, "2024-04")
    run_month(env, "2024-03")
    with env[1].connect() as conn:
        assert count_batch(conn, "clean_posts", "2024-03") == c_march["clean_posts"]
        assert count_batch(conn, "clean_posts", "2024-04") > 0


def test_quality_gate_passes_on_good_data(env):
    raw, engine = env
    start, end, batch = P.batch_bounds("2024-03")
    P.extract(raw, engine, start, end, batch)
    P.validate(raw, engine, batch)
    P.transform(raw, engine, start, end, batch)
    result = P.quality_check(raw, engine, start, end, batch)
    assert "empty_text_share" in result


def test_quality_gate_fails_on_a_broken_batch_and_nothing_is_loaded(tmp_path):
    bars = make_bars(days=120)
    posts = make_posts(bars, n=800)
    # break it on purpose: blank out 80% of the texts
    posts.loc[posts.index % 5 != 0, "text"] = ""
    raw = write_raw(tmp_path, bars, posts)
    engine = create_engine(f"sqlite:///{tmp_path}/wh.db")
    with pytest.raises(QualityGateError, match="empty_text_share"):
        run_month((raw, engine), "2024-03")
    with engine.connect() as conn:
        assert count_batch(conn, "clean_posts", "2024-03") == 0


def test_validate_rejects_a_changed_schema(tmp_path):
    bars = make_bars(days=120)
    posts = make_posts(bars, n=200).rename(columns={"text": "content"})
    raw = write_raw(tmp_path, bars, posts)
    start, end, batch = P.batch_bounds("2024-03")
    engine = create_engine(f"sqlite:///{tmp_path}/wh.db")
    with pytest.raises((SchemaError, KeyError, ValueError)):
        P.extract(raw, engine, start, end, batch)
        P.validate(raw, engine, batch)


def test_features_are_point_in_time(env):
    raw, engine = env
    _, _, batch, *_ = run_month(env, "2024-03")
    info = P.build_features(engine, raw, batch)
    assert info["feature_rows"] > 0 and info["feature_version"] == "v1"

    feats = pd.read_sql(
        text("SELECT * FROM features WHERE batch = :b"), engine, params={"b": batch}
    )
    feats["event_timestamp"] = pd.to_datetime(feats["event_timestamp"], utc=True)
    all_posts = load_posts(Path(raw) / "train-00000-of-00001.parquet")
    row = feats.iloc[len(feats) // 2]
    t = row["event_timestamp"]
    created = all_posts["created_at"]
    manual = ((created >= t - pd.Timedelta("60min")) & (created < t)).sum()
    assert row["posts_prev_60min"] == manual
    # the last completed bar ended before the post's own bar started
    assert feats["last_bar_return"].notna().all()


def test_tasks_hand_over_data_through_the_warehouse_and_return_only_metadata(env):
    """The orchestrator carries counters, the data stays in the warehouse (stg_* tables)."""
    import json

    raw, engine = env
    start, end, batch = P.batch_bounds("2024-03")
    returned = [
        P.extract(raw, engine, start, end, batch),
        P.validate(raw, engine, batch),
        P.transform(raw, engine, start, end, batch),
        P.quality_check(raw, engine, start, end, batch),
    ]
    for x in returned:
        assert len(json.dumps(x, default=str)) < 2_000  # counters and strings, no rows
    with engine.connect() as conn:
        assert count_batch(conn, "stg_posts", batch) == returned[0]["posts_in_batch"] > 0
        assert count_batch(conn, "stg_clean", batch) == returned[2]["clean_rows"] > 0
        assert count_batch(conn, "clean_posts", batch) == 0  # nothing loaded before `load`


def test_the_dag_declares_the_order_with_shift_operators_and_passes_no_results():
    source = (Path(__file__).parents[1] / "dags" / "truth_social_etl.py").read_text()
    flat = "".join(source.split())  # the chain is split over several lines
    order = ["extract", "validate", "transform", "quality_check", "load", "build_features"]
    assert ">>".join(f"{t}()" for t in order + ["materialize"]) in flat
    assert "load(quality_check(" not in source and "(x: dict)" not in source


def test_materialize_returns_what_the_feast_script_printed(tmp_path):
    repo = tmp_path / "feature_repo"
    repo.mkdir()
    (repo / "materialize.py").write_text(
        'print("noise")\nprint(\'{"online_slots": 5, "missing_slots": []}\')\n'
    )
    got = P.materialize_online(sys.executable, repo)
    assert got == {"online_slots": 5, "missing_slots": []}


def test_materialize_fails_loudly_when_the_feast_script_fails(tmp_path):
    repo = tmp_path / "feature_repo"
    repo.mkdir()
    (repo / "materialize.py").write_text('raise SystemExit("online store has no sigma")\n')
    with pytest.raises(RuntimeError, match="no sigma"):
        P.materialize_online(sys.executable, repo)
