"""Make a deliberately broken copy of the raw data to show the quality gate failing.

Copies data/raw to data/raw_bad and blanks the text of every post of one month, as if the
source had started to send media-only or repost-only items. The gate must then fail on the
share of empty texts. The original data/raw is not touched.

    uv run python scripts/make_bad_batch.py 2025-03
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POSTS = "train-00000-of-00001.parquet"


def main(month: str) -> None:
    src, dst = ROOT / "data" / "raw", ROOT / "data" / "raw_bad"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    df = pd.read_parquet(dst / POSTS)
    ts = pd.to_datetime(df["datetime"], utc=True)
    in_month = ts.dt.strftime("%Y-%m") == month
    df.loc[in_month, "text"] = ""
    df.to_parquet(dst / POSTS, index=False)
    print(f"blanked {int(in_month.sum())} texts of {month} in {dst}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2025-03")
