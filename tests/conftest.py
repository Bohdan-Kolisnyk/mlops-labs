"""Synthetic raw data with the same schema as the real files, so tests need no downloads."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_bars(days: int = 60, seed: int = 0, start: str = "2024-01-02") -> pd.DataFrame:
    """Seven hourly bars per weekday (09:30 ... 15:30 ET), as the yfinance CSV delivers them."""
    rng = np.random.default_rng(seed)
    rows = []
    price = 4500.0
    for day in pd.bdate_range(start, periods=days):
        for hour in range(7):
            ts = (day + pd.Timedelta(hours=9, minutes=30 + 60 * hour)).tz_localize(
                "America/New_York"
            )
            o = price
            c = o * float(np.exp(rng.normal(0, 0.002)))
            rows.append((ts.tz_convert("UTC"), o, max(o, c), min(o, c), c, 1_000))
            price = c
    df = pd.DataFrame(rows, columns=["bar_start", "open", "high", "low", "close", "volume"])
    et = df["bar_start"].dt.tz_convert("America/New_York")
    df["session_date"] = et.dt.normalize().dt.tz_localize(None)
    df["slot"] = et.dt.strftime("%H:%M")
    return df


@pytest.fixture()
def bars() -> pd.DataFrame:
    return make_bars()


def write_raw(tmp_path, bars_df: pd.DataFrame, posts: pd.DataFrame) -> str:
    """Write the two raw files in the formats of the real ones and return the directory."""
    raw = tmp_path / "raw"
    raw.mkdir()
    posts.to_parquet(raw / "train-00000-of-00001.parquet", index=False)
    et = bars_df["bar_start"].dt.tz_convert("America/New_York")
    csv = pd.DataFrame(
        {
            "Datetime": et.astype(str),
            "Adj Close": bars_df["close"],
            "Close": bars_df["close"],
            "High": bars_df["high"],
            "Low": bars_df["low"],
            "Open": bars_df["open"],
            "Volume": bars_df["volume"],
        }
    )
    with open(raw / "gspc_1h.csv", "w", newline="") as f:
        f.write("Price,Adj Close,Close,High,Low,Open,Volume\n")
        f.write("Ticker,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC\n")
        f.write("Datetime,,,,,,\n")
        csv.to_csv(f, header=False, index=False)
    return str(raw)


def make_posts(bars_df: pd.DataFrame, n: int = 400, seed: int = 1) -> pd.DataFrame:
    """Posts spread over the trading days of `bars_df`, some empty, a few repeated."""
    rng = np.random.default_rng(seed)
    starts = list(bars_df["bar_start"])
    rows = []
    for i in range(n):
        base = starts[int(rng.integers(0, len(starts)))]
        ts = base + pd.Timedelta(seconds=int(rng.integers(0, 3600)))
        text = "" if rng.random() < 0.2 else f"post number {i % 350} about tariff"
        rows.append((str(i), ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"), text, bool(rng.random() < 0.1)))
    return pd.DataFrame(rows, columns=["post_id", "datetime", "text", "has_media"])
