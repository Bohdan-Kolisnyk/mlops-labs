"""Write the offline data of the Feast feature view from the versioned raw data.

    uv run python scripts/export_slot_sigma.py

Writes into feature_repo/data/ (derived, git-ignored):
  slot_sigma.parquet  (also uploaded to MinIO, s3://lake/feast/, the offline store of Feast)
                      one row per usable hourly bar: slot, event_timestamp (bar start, UTC),
                      sigma (std of the same two-bar return in the same slot over the previous
                      20 trading days; only past values, so it is known when the bar starts);
  post_check.parquet  post_id, post time, slot and the sigma that the training table holds for
                      the post, used by feature_repo/demo.py to show that serving and training
                      agree.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops_labs import config, dataset
from mlops_labs.data import load_bars
from mlops_labs.env import ROOT, load_env
from mlops_labs.labels import build_windows

OUT = ROOT / "feature_repo" / "data"
S3_URI = "s3://lake/feast/slot_sigma.parquet"


def s3_options() -> dict:
    """MinIO endpoint and keys, from the same .env as docker compose."""
    return {
        "key": os.environ["MINIO_ROOT_USER"],
        "secret": os.environ["MINIO_ROOT_PASSWORD"],
        "client_kwargs": {
            "endpoint_url": os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
        },
    }


def main() -> None:
    load_env()
    raw = ROOT / "data" / "raw"
    OUT.mkdir(parents=True, exist_ok=True)

    windows = build_windows(load_bars(raw / config.BARS_FILE))
    sigma = windows[windows["slot"].isin(config.POST_SLOTS) & windows["sigma"].notna()]
    sigma = sigma[["slot", "bar_start", "sigma"]].rename(columns={"bar_start": "event_timestamp"})
    sigma = sigma.sort_values(["slot", "event_timestamp"]).reset_index(drop=True)
    sigma.to_parquet(OUT / "slot_sigma.parquet", index=False)
    sigma.to_parquet(S3_URI, index=False, storage_options=s3_options())

    table, data_version = dataset.from_raw(raw)
    check = table[["post_id", "created_at", "sigma", "slot_index"]].copy()
    check["slot"] = check["slot_index"].astype(int).map(dict(enumerate(config.POST_SLOTS)))
    check = check.rename(columns={"created_at": "event_timestamp", "sigma": "sigma_expected"})
    check[["post_id", "slot", "event_timestamp", "sigma_expected"]].to_parquet(
        OUT / "post_check.parquet", index=False
    )

    (OUT / "export.json").write_text(
        json.dumps(
            {
                "data_version": data_version,
                "slot_sigma_rows": len(sigma),
                "post_check_rows": len(check),
            },
            indent=2,
        )
    )
    print(f"data version {data_version}: {len(sigma)} sigma rows, {len(check)} posts -> {OUT}")
    print(f"sigma rows also in the offline store: {S3_URI}")


if __name__ == "__main__":
    main()
