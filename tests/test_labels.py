import numpy as np
import pandas as pd

from mlops_labs import config
from mlops_labs.labels import build_windows, label_posts
from tests.conftest import make_bars


def test_window_return_uses_next_bar_of_same_session(bars):
    w = build_windows(bars)
    b = bars.reset_index(drop=True)
    i = 0  # first bar of the first day
    expected = np.log(b.loc[i + 1, "close"] / b.loc[i, "open"])
    assert np.isclose(w.loc[i, "r"], expected)
    # the last bar of a day has no full next bar in the same session
    last = b.index[b["slot"] == "15:30"][0]
    assert np.isnan(w.loc[last, "r"])


def test_sigma_uses_only_past_days():
    bars = make_bars(days=40)
    w = build_windows(bars)
    slot = w[w["slot"] == "10:30"].reset_index(drop=True)
    day = 25
    past = slot.loc[day - config.SIGMA_DAYS : day - 1, "r"]
    assert np.isclose(slot.loc[day, "sigma"], past.std())
    # not defined before 20 past days exist
    assert slot.loc[: config.SIGMA_DAYS - 1, "sigma"].isna().all()


def test_label_changes_only_with_the_threshold(bars):
    loose = build_windows(bars, k=0.5)["y"].dropna().astype(int).sum()
    strict = build_windows(bars, k=2.5)["y"].dropna().astype(int).sum()
    assert loose > strict


def test_posts_outside_the_window_are_dropped(bars):
    w = build_windows(bars)
    day = bars["session_date"].iloc[30 * 7]
    et = pd.Timestamp(day).tz_localize("America/New_York")

    def post(i, hour, minute):
        t = (et + pd.Timedelta(hours=hour, minutes=minute)).tz_convert("UTC")
        return {"post_id": str(i), "text": f"t{i}", "has_media": False, "created_at": t}

    posts = pd.DataFrame(
        [post(1, 9, 31), post(2, 14, 29), post(3, 14, 31), post(4, 17, 0), post(5, 8, 0)]
    )
    labelled = label_posts(posts, w)
    assert set(labelled["post_id"]) == {"1", "2"}


def test_empty_and_duplicate_flags(bars):
    w = build_windows(bars)
    day = bars["session_date"].iloc[30 * 7]
    et = pd.Timestamp(day).tz_localize("America/New_York")
    times = [(et + pd.Timedelta(hours=10, minutes=m)).tz_convert("UTC") for m in (0, 5, 10)]
    posts = pd.DataFrame(
        {
            "post_id": ["a", "b", "c"],
            "text": ["same", "same", "  "],
            "has_media": False,
            "created_at": times,
        }
    )
    out = label_posts(posts, w).set_index("post_id")
    assert not out.loc["a", "dup_text"]
    assert out.loc["b", "dup_text"]
    assert out.loc["c", "text_empty"] and not out.loc["c", "dup_text"]
