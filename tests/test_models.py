import numpy as np
import pandas as pd

from mlops_labs import dataset, models


def test_keyword_rule_counts_whole_words_only():
    df = pd.DataFrame(
        {"text": ["Tariffs on China are coming", "marketing and oils", "", "the Fed and the Fed"]}
    )
    assert list(models.KeywordRule().score(df)) == [2, 0, 0, 2]


def test_split_is_by_session_date_and_has_no_overlap():
    d = pd.to_datetime(
        ["2024-01-05", "2025-06-30", "2025-07-01", "2025-12-31", "2026-01-05", "2026-10-16"]
    )
    out = dataset.split_name(pd.Series(d))
    assert list(out) == ["train", "train", "val", "val", "test", "test"]


def test_alert_rate_is_exactly_the_top_rate_even_when_every_score_ties():
    n = 1000
    y = (np.random.default_rng(0).random(n) < 0.15).astype(int)
    scores = np.full(n, 0.15)
    threshold = models.alert_threshold(scores, seed=0)
    op = models.operating_point(y, scores, threshold, seed=0)
    assert abs(op["alert_rate"] - models.TOP_RATE) < 0.01


def test_operating_point_catches_what_a_perfect_score_ranks_first():
    y = np.array([1] * 10 + [0] * 90)
    scores = y.astype(float)  # perfect ranking
    threshold = models.alert_threshold(scores, seed=0)
    op = models.operating_point(y, scores, threshold, seed=0)
    assert op["coverage"] == 1.0
    assert abs(op["precision"] - 0.5) < 1e-9  # 10 hits among 20 alerts


def test_bootstrap_resamples_whole_hours_and_pairs_the_models():
    rng = np.random.default_rng(1)
    hours = np.repeat(np.arange(100), 3)
    y = (rng.random(300) < 0.2).astype(int)
    good = y + rng.normal(0, 0.3, 300)
    noise = rng.random(300)
    ci = models.clustered_bootstrap_ci(
        y, {"good": good, "noise": noise}, hours, n=200, seed=0, reference="noise"
    )
    low, high = ci["good-noise"]
    assert 0 < low < high  # the informative score is better than noise on every resample range
    assert ci["noise"][0] < ci["noise"][1]


def test_context_model_and_candidate_fit_and_score_a_tiny_table():
    rng = np.random.default_rng(2)
    n = 300
    df = pd.DataFrame(
        {
            "text": rng.choice(["tariff news today", "nice weather", "fed meeting", "hello"], n),
            "y": (rng.random(n) < 0.2).astype(int),
            "text_len": rng.integers(5, 200, n),
            "link_count": rng.integers(0, 2, n),
            "caps_share": rng.random(n),
            "has_media": rng.integers(0, 2, n),
            "slot_index": rng.integers(0, 5, n),
            "weekday": rng.integers(0, 5, n),
            "minutes_since_open": rng.integers(0, 300, n),
            "sigma": rng.random(n) * 0.005,
            "last_bar_return": rng.normal(0, 0.002, n),
            "posts_prev_60min": rng.integers(0, 6, n),
        }
    )
    df.loc[0, "last_bar_return"] = np.nan  # the first bar of the archive has no predecessor
    for model in (models.ConstantPrior(), models.KeywordRule(), models.ContextOnly()):
        s = model.fit(df).score(df)
        assert len(s) == n and np.isfinite(s).all()
    s = models.Candidate(c=1.0).fit(df).score(df)
    assert len(s) == n and ((0 <= s) & (s <= 1)).all()


def test_the_vocabulary_is_cut_by_a_threshold_not_by_a_top_k():
    # a top-k cut breaks ties with a platform-dependent sort, see build_candidate
    tfidf = models.build_candidate().named_steps["prep"].transformers[0][1]
    assert tfidf.max_features is None and tfidf.min_df == 5
