"""Feature table (design 5.3). Every feature is computable at serving time from the
request, the price feed, the online store or the recent posts table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlops_labs import config

FEATURE_COLUMNS = [
    "text_len",
    "link_count",
    "caps_share",
    "has_media",
    "slot_index",
    "weekday",
    "minutes_since_open",
    "sigma",
    "last_bar_return",
    "posts_prev_60min",
]


def _naive_utc(s: pd.Series) -> np.ndarray:
    """UTC timestamps as a numpy datetime64[ns] array (for searchsorted arithmetic)."""
    return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")


def compute_features(
    clean: pd.DataFrame, all_posts: pd.DataFrame, bars: pd.DataFrame
) -> pd.DataFrame:
    """Features for the usable, non-empty, non-duplicate posts of `clean`.

    Point-in-time rules: `last_bar_return` is the return of the bar that ended before the
    post's bar started, and `posts_prev_60min` counts only posts strictly before the post.
    """
    c = clean[~clean["text_empty"] & ~clean["dup_text"]].copy()
    et = c["created_at"].dt.tz_convert(config.ET)

    letters = c["text"].str.count(r"[A-Za-z]")
    caps = c["text"].str.count(r"[A-Z]")

    b = bars.sort_values("bar_start").reset_index(drop=True)
    prev_ret = np.log(b["close"] / b["open"]).shift(1)
    prev_ret.index = b["bar_start"]

    stamps = _naive_utc(all_posts["created_at"].dropna().sort_values())
    t = _naive_utc(c["created_at"])
    window = np.timedelta64(60, "m")
    n_prev = np.searchsorted(stamps, t, side="left") - np.searchsorted(
        stamps, t - window, side="left"
    )

    out = pd.DataFrame(
        {
            "post_id": c["post_id"].to_numpy(),
            "event_timestamp": c["created_at"].reset_index(drop=True),
            "feature_version": config.FEATURE_VERSION,
            "text_len": c["text"].str.len().to_numpy(),
            "link_count": c["text"].str.count(r"https?://").to_numpy(),
            "caps_share": (caps / letters.where(letters > 0)).fillna(0.0).to_numpy(),
            "has_media": c["has_media"].astype(int).to_numpy(),
            "slot_index": c["slot"].map({s: i for i, s in enumerate(config.POST_SLOTS)}).to_numpy(),
            "weekday": et.dt.day_of_week.to_numpy(),
            "minutes_since_open": (et.dt.hour * 60 + et.dt.minute - 570).to_numpy(),
            "sigma": c["sigma"].to_numpy(),
            "last_bar_return": c["bar_start"].map(prev_ret).to_numpy(),
            "posts_prev_60min": n_prev,
        }
    )
    return out.reset_index(drop=True)
