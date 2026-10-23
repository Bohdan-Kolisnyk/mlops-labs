"""The label of design 5.1: an abnormal two-bar move of the index after a post."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlops_labs import config


def build_windows(bars: pd.DataFrame, k: float = config.K) -> pd.DataFrame:
    """One row per hourly bar c that is followed by a full bar c+1 in the same session.

    r     = log(close(c+1) / open(c))
    sigma = std of r at the same time slot over the previous SIGMA_DAYS trading days
            (only past values: the current day is never part of its own sigma)
    y     = 1 if |r| > k * sigma, 0 otherwise, missing while sigma is not defined
    """
    b = bars.sort_values("bar_start").reset_index(drop=True)
    next_same_session = b["session_date"].shift(-1) == b["session_date"]
    next_is_full_hour = (b["bar_start"].shift(-1) - b["bar_start"]) == pd.Timedelta("60min")
    r = np.log(b["close"].shift(-1) / b["open"]).where(next_same_session & next_is_full_hour)

    w = pd.DataFrame(
        {
            "bar_start": b["bar_start"],
            "session_date": b["session_date"],
            "slot": b["slot"],
            "r": r,
        }
    )

    def past_sigma(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        sig = valid.shift(1).rolling(config.SIGMA_DAYS, min_periods=config.SIGMA_DAYS).std()
        return sig.reindex(s.index)

    w["sigma"] = w.groupby("slot")["r"].transform(past_sigma)
    defined = w["r"].notna() & w["sigma"].notna()
    w["y"] = (w["r"].abs() > k * w["sigma"]).where(defined).astype("Int64")
    return w


def label_posts(posts: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Attach bar c, r, sigma and y to every usable post.

    Usable: the post lies inside a bar whose start slot is in POST_SLOTS and the label is
    defined. Also flags the two cleaning rules (design 5.2, known data issues):
      text_empty - the post has no text (media or a repost);
      dup_text   - the same text was already posted earlier among the usable posts.
    """
    p = posts.dropna(subset=["created_at"]).sort_values("created_at", kind="stable")
    w = windows[["bar_start", "session_date", "slot", "r", "sigma", "y"]].sort_values("bar_start")
    m = pd.merge_asof(
        p,
        w,
        left_on="created_at",
        right_on="bar_start",
        direction="backward",
        tolerance=pd.Timedelta("60min"),
    )
    m = m[m["bar_start"].notna() & m["slot"].isin(config.POST_SLOTS) & m["y"].notna()].copy()
    m["y"] = m["y"].astype(int)
    m["text_empty"] = m["text"].str.strip() == ""
    m = m.sort_values("created_at", kind="stable")
    m["dup_text"] = (~m["text_empty"]) & m.duplicated("text", keep="first")
    return m.reset_index(drop=True)
