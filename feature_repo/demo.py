"""Point-in-time retrieval demo (Lab 2, step 4). Run after `feast apply`:

cd feature_repo
uv run feast apply
uv run python demo.py
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
from feast import FeatureStore

HERE = Path(__file__).resolve().parent
FEATURE = "slot_sigma:sigma"


def main() -> None:
    store = FeatureStore(repo_path=str(HERE))
    rows = pd.read_parquet(HERE / "data" / "slot_sigma.parquet")

    # 1. entity timestamps between two feature rows return the earlier row
    one = rows[rows["slot"] == "10:30"].sort_values("event_timestamp").reset_index(drop=True)
    first, earlier, later = one.iloc[0], one.iloc[100], one.iloc[101]
    asked = pd.DataFrame(
        {
            "slot": ["10:30"] * 3,
            "event_timestamp": [
                first["event_timestamp"] - timedelta(days=1),  # before any row existed
                earlier["event_timestamp"] + timedelta(hours=11),  # between two rows
                later["event_timestamp"],  # exactly at a row
            ],
        }
    )
    got = store.get_historical_features(entity_df=asked, features=[FEATURE]).to_df()
    got = got.sort_values("event_timestamp").reset_index(drop=True)
    print("feature rows of slot 10:30 around the asked times")
    print(f"  earlier row: {earlier['event_timestamp']}  sigma = {earlier['sigma']:.6f}")
    print(f"  later row:   {later['event_timestamp']}  sigma = {later['sigma']:.6f}\n")
    print("asked at:")
    print(asked.to_string(index=False), "\n")
    print("returned:")
    print(got[["slot", "event_timestamp", "sigma"]].to_string(index=False))
    assert len(got) == 2, "the time before the first row has no feature value and is not returned"
    assert got.loc[0, "sigma"] == earlier["sigma"], "between two rows the earlier row must win"
    assert got.loc[1, "sigma"] == later["sigma"], "exactly at a row the row itself is returned"
    print(
        "\nOK: nothing before the first row, the earlier row between two rows, "
        "the row itself at its own time"
    )

    # 2. the store returns what the training table holds, for every post
    check = pd.read_parquet(HERE / "data" / "post_check.parquet")
    served = store.get_historical_features(
        entity_df=check[["post_id", "slot", "event_timestamp"]], features=[FEATURE]
    ).to_df()
    merged = check.merge(served[["post_id", "sigma"]], on="post_id", validate="one_to_one")
    diff = (merged["sigma"] - merged["sigma_expected"]).abs()
    print(
        f"\n{len(merged)} posts: missing {int(merged['sigma'].isna().sum())}, "
        f"max |served - training| = {diff.max():.1e}"
    )
    assert merged["sigma"].notna().all() and diff.max() < 1e-12

    # 3. online side: materialize, then ask for the latest sigma of a slot
    last = rows["event_timestamp"].max()
    store.materialize(
        start_date=(rows["event_timestamp"].min() - timedelta(days=1)).to_pydatetime(),
        end_date=(last + timedelta(hours=1)).to_pydatetime(),
    )
    online = store.get_online_features(
        features=[FEATURE], entity_rows=[{"slot": "10:30"}]
    ).to_dict()
    newest = one.iloc[-1]
    print(
        f"\nonline sigma of slot 10:30: {online['sigma'][0]:.6f} "
        f"(newest row {newest['event_timestamp']}: {newest['sigma']:.6f})"
    )
    assert online["sigma"][0] == newest["sigma"]
    print("OK: the online store holds the newest row of each slot")


if __name__ == "__main__":
    main()
