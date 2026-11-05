"""Fill the online store (PostgreSQL) with the newest sigma of each slot.

    cd feature_repo && uv run python materialize.py

Reads the offline store (parquet in MinIO) and writes to the online store. Incremental: only
the period since the previous run is read. The last line is a JSON with the result, which the
Airflow task `materialize` returns as its (small) XCom.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from feast import FeatureStore
from feast_env import setup_env

HERE = Path(__file__).resolve().parent
FEATURE = "slot_sigma:sigma"
SLOTS = ["09:30", "10:30", "11:30", "12:30", "13:30"]


def main() -> None:
    setup_env()
    store = FeatureStore(repo_path=str(HERE))
    store.materialize_incremental(end_date=datetime.now(UTC))
    online = store.get_online_features(
        features=[FEATURE], entity_rows=[{"slot": s} for s in SLOTS]
    ).to_dict()
    sigma = {s: v for s, v in zip(online["slot"], online["sigma"], strict=True)}
    missing = [s for s, v in sigma.items() if v is None]
    print(json.dumps({"online_slots": len(sigma) - len(missing), "missing_slots": missing}))
    if missing:
        raise SystemExit(f"online store has no sigma for slots {missing}")


if __name__ == "__main__":
    main()
