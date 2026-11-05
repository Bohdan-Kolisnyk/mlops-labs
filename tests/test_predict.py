"""The serving path (text + time -> features -> score) must agree with the training table."""

import numpy as np
import pandas as pd
import pytest

from mlops_labs import config, dataset, models, predict
from mlops_labs.data import load_bars, load_posts
from mlops_labs.labels import build_windows
from tests.conftest import make_bars, make_posts, write_raw


class FakeModel:
    """Stands in for the registry model: a probability that grows with sigma."""

    def predict_proba(self, x):
        p = 1.0 / (1.0 + np.exp(-300.0 * (x["sigma"].to_numpy() - 0.002)))
        return np.column_stack([1 - p, p])


@pytest.fixture()
def world(tmp_path, monkeypatch):
    bars = make_bars(days=120)
    posts = make_posts(bars, n=900)
    raw = write_raw(tmp_path, bars, posts)
    windows = build_windows(load_bars(f"{raw}/{config.BARS_FILE}"))

    def fake_feature_store(requests, online=False):
        """sigma of the bar at or before `at` in the same slot, like the point-in-time join."""
        out = []
        for r in requests:
            rows = windows[(windows["slot"] == r["slot"]) & windows["sigma"].notna()]
            rows = rows[rows["bar_start"] <= pd.Timestamp(r["at"])]
            out.append(None if rows.empty else float(rows.iloc[-1]["sigma"]))
        return out

    monkeypatch.setattr(predict, "fetch_sigma", fake_feature_store)
    return raw


def test_serving_features_equal_the_training_table(world):
    raw = world
    table, _ = dataset.from_raw(raw)
    sample = table.sample(40, random_state=0)
    posts = load_posts(f"{raw}/{config.POSTS_FILE}")
    asked = posts[posts["post_id"].isin(sample["post_id"])]
    bars = load_bars(f"{raw}/{config.BARS_FILE}")
    feats, skipped = predict.serving_features(asked, posts[["post_id", "created_at"]], bars)
    assert skipped.empty
    got = feats.set_index("post_id")[models.FEATURE_COLUMNS].loc[sample["post_id"]]
    want = sample.set_index("post_id")[models.FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(got, want, check_dtype=False)


def test_the_order_of_the_posts_does_not_change_their_scores(world):
    raw = world
    posts = load_posts(f"{raw}/{config.POSTS_FILE}")
    table, _ = dataset.from_raw(raw)  # posts whose label, and so sigma, is defined
    asked = posts[posts["post_id"].isin(table["post_id"].iloc[:70])]
    asked = asked[["post_id", "created_at", "text", "has_media"]]
    model = FakeModel()
    a, _ = predict.score_posts(asked, raw, model=model, threshold=0.5)
    b, _ = predict.score_posts(
        asked.sample(frac=1, random_state=1), raw, model=model, threshold=0.5
    )
    a, b = a.set_index("post_id").sort_index(), b.set_index("post_id").sort_index()
    pd.testing.assert_series_equal(a["probability"], b["probability"])
    assert a["probability"].notna().sum() > 0


def test_posts_outside_the_window_or_without_text_are_not_scored_and_say_why(world):
    raw = world
    posts = pd.DataFrame(
        {
            "post_id": ["sat", "night", "empty", "ok"],
            "created_at": pd.to_datetime(
                [
                    "2024-03-16T15:00:00Z",  # Saturday
                    "2024-03-14T03:00:00Z",  # before the open
                    "2024-03-14T15:00:00Z",  # empty text
                    "2024-03-14T15:00:00Z",
                ],
                utc=True,
            ),
            "text": ["tariffs", "tariffs", "  ", "tariffs on china"],
            "has_media": [False, False, True, False],
        }
    )
    out, _ = predict.score_posts(posts, raw, model=FakeModel(), threshold=0.5)
    out = out.set_index("post_id")
    assert out.loc["sat", "reason"] == predict.OUT_OF_WINDOW
    assert out.loc["night", "reason"] == predict.OUT_OF_WINDOW
    assert out.loc["empty", "reason"] == "no text (media or repost)"
    assert out.loc["ok", "reason"] == "" and 0 < out.loc["ok", "probability"] < 1


def test_the_alert_flag_is_the_comparison_with_the_threshold(world):
    raw = world
    posts = load_posts(f"{raw}/{config.POSTS_FILE}")[["post_id", "created_at", "text", "has_media"]]
    out, info = predict.score_posts(posts.iloc[:200], raw, model=FakeModel(), threshold=0.5)
    scored = out[out["reason"] == ""]
    assert (scored["alert"] == (scored["probability"] >= 0.5)).all()
    assert info["alert_threshold"] == 0.5
