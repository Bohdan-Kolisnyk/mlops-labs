"""sigma of a time slot from the feature store, for scripts/predict.py (runs in this environment).

stdin : a JSON list of {"slot": "10:30", "at": "2025-03-12T15:05:00+00:00"}
stdout: one JSON line, a list with one sigma per request (null when the store has none)

Default is the offline store with a point-in-time join: the row valid at `at`, the same value
the training table holds. With --online the newest value of each slot in the online store is
returned instead (what a live service reads; `at` is ignored).

    cd feature_repo && uv run python apply.py     # once; the online store is filled by materialize.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from feast import FeatureStore
from feast_env import setup_env

HERE = Path(__file__).resolve().parent
FEATURE = "slot_sigma:sigma"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()

    setup_env()
    asked = json.load(sys.stdin)
    store = FeatureStore(repo_path=str(HERE))

    if args.online:
        out = store.get_online_features(
            features=[FEATURE], entity_rows=[{"slot": a["slot"]} for a in asked]
        ).to_dict()
        sigmas = [None if v is None else float(v) for v in out["sigma"]]
    else:
        entity_df = pd.DataFrame(
            {
                "slot": [a["slot"] for a in asked],
                "event_timestamp": pd.to_datetime([a["at"] for a in asked], utc=True),
                "row": range(len(asked)),
            }
        )
        got = store.get_historical_features(entity_df=entity_df, features=[FEATURE]).to_df()
        by_row = dict(zip(got["row"], got["sigma"], strict=True))
        sigmas = [None if pd.isna(by_row.get(i)) else float(by_row[i]) for i in range(len(asked))]

    print(json.dumps(sigmas))


if __name__ == "__main__":
    main()
