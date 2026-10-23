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
    return raw, str(tmp_path / "staging"), engine


def run_month(env, month):
    raw, staging, engine = env
    start, end, batch = P.batch_bounds(month)
    P.extract(raw, staging, start, end, batch)
    P.validate(raw, staging, batch)
    stats = P.transform(raw, staging, start, end, batch)
    P.quality_check(raw, staging, start, end, batch)
    counts = P.load(engine, staging, batch)
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
    clean = P.read_clean_batch(env[2], batch)
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
    with env[2].connect() as conn:
        assert count_batch(conn, "clean_posts", "2024-03") == c_march["clean_posts"]
        assert count_batch(conn, "clean_posts", "2024-04") > 0


def test_quality_gate_passes_on_good_data(env):
    raw, staging, _ = env
    start, end, batch = P.batch_bounds("2024-03")
    P.extract(raw, staging, start, end, batch)
    P.validate(raw, staging, batch)
    P.transform(raw, staging, start, end, batch)
    result = P.quality_check(raw, staging, start, end, batch)
    assert "empty_text_share" in result


def test_quality_gate_fails_on_a_broken_batch_and_nothing_is_loaded(tmp_path):
    bars = make_bars(days=120)
    posts = make_posts(bars, n=800)
    # break it on purpose: blank out 80% of the texts
    posts.loc[posts.index % 5 != 0, "text"] = ""
    raw = write_raw(tmp_path, bars, posts)
    staging = str(tmp_path / "staging")
    engine = create_engine(f"sqlite:///{tmp_path}/wh.db")
    with pytest.raises(QualityGateError, match="empty_text_share"):
        run_month((raw, staging, engine), "2024-03")
    with engine.connect() as conn:
        assert count_batch(conn, "clean_posts", "2024-03") == 0


def test_validate_rejects_a_changed_schema(tmp_path):
    bars = make_bars(days=120)
    posts = make_posts(bars, n=200).rename(columns={"text": "content"})
    raw = write_raw(tmp_path, bars, posts)
    start, end, batch = P.batch_bounds("2024-03")
    with pytest.raises((SchemaError, KeyError, ValueError)):
        P.extract(raw, str(tmp_path / "st"), start, end, batch)
        P.validate(raw, str(tmp_path / "st"), batch)


def test_features_are_point_in_time(env):
    raw, _, engine = env
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
