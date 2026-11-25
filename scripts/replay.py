"""Replay held-out posts against the ML service, optionally with an injected distribution shift.

    uv run --with requests --with badgers python scripts/replay.py --mode baseline
    uv run --with requests --with badgers python scripts/replay.py --mode shift

Rows come from the seed (deploy/seed/train_table.parquet): by default the training rows outside the
reference sample, which are in distribution; with --source test the held-out 2026 rows, which the
model never saw and which drift by themselves. `baseline` sends them unchanged. `shift` first applies one documented transformation made with
Badgers (FixedShiftGenerator): the chosen columns move by `--shift-std` standard deviations of the
batch, in exactly `--fraction` of the rows. Every row goes through POST /predict, so the service
logs it: the request log is what the drift job reads. With --labels the true labels follow through
POST /feedback, as they would arrive later in production.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
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
INTEGER = {"text_len", "link_count", "posts_prev_60min", "minutes_since_open"}


def shifted(df: pd.DataFrame, columns: list[str], std: float, fraction: float, seed: int) -> dict:
    from badgers.core.selectors import CountSelector
    from badgers.generators.tabular_data.drift import FixedShiftGenerator

    rng = np.random.default_rng(seed)
    x = df[columns].to_numpy(dtype=float)
    count = round(fraction * len(df))
    selector = CountSelector(columns=list(range(len(columns))), count=count, rng=rng)
    y = FixedShiftGenerator(rng).generate(x.copy(), shift=std, selector=selector)
    out = df.copy()
    changed = np.abs(y - x).max(axis=1) > 1e-12
    for i, col in enumerate(columns):
        values = y[:, i]
        if col == "caps_share":
            values = np.clip(values, 0.0, 1.0)
        elif col == "sigma":
            values = np.clip(values, 1e-6, None)
        elif col in INTEGER:
            values = np.clip(np.round(values), 0, None)
        out[col] = values
    return {
        "frame": out,
        "description": {
            "generator": "badgers FixedShiftGenerator + CountSelector",
            "columns": columns,
            "shift_in_batch_std": std,
            "fraction_of_rows": fraction,
            "rows_affected": int(changed.sum()),
            "rows_total": len(df),
            "mean_before": {c: round(float(df[c].mean()), 6) for c in columns},
            "mean_after": {c: round(float(out[c].mean()), 6) for c in columns},
        },
    }


def payload(df: pd.DataFrame) -> list[dict]:
    cols = ["text", *FEATURES]
    clean = df[cols].astype(object).where(df[cols].notna(), None)
    rows = clean.to_dict("records")
    for r in rows:
        r["text"] = str(r["text"])
        for f in FEATURES:
            if r[f] is not None:
                r[f] = float(r[f])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--mode", choices=["baseline", "shift"], required=True)
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--offset", type=int, default=None, help="first row of the shuffled test part")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument(
        "--source",
        choices=["train-rest", "test"],
        default="train-rest",
        help="train-rest: training rows outside the 1,000-row reference sample (in distribution);\n"
        "test: the held-out 2026 rows, which drift naturally (design 7.3)",
    )
    ap.add_argument("--shift-columns", default="sigma")
    ap.add_argument("--shift-std", type=float, default=3.0)
    ap.add_argument("--fraction", type=float, default=0.4)
    ap.add_argument("--labels", action="store_true", help="send the true labels afterwards")
    ap.add_argument("--seed-file", default=str(ROOT / "deploy" / "seed" / "train_table.parquet"))
    args = ap.parse_args()

    table = pd.read_parquet(args.seed_file)
    if args.source == "test":
        pool = table[table["split"] == "test"]
    else:
        train = table[table["split"] == "train"]
        pool = train.drop(train.sample(n=min(1000, len(train)), random_state=0).index)
    test = pool.sample(frac=1.0, random_state=args.seed)
    start = args.offset if args.offset is not None else (0 if args.mode == "baseline" else args.rows)
    rows = test.iloc[start : start + args.rows].reset_index(drop=True)
    if len(rows) < args.rows:
        sys.exit(f"only {len(rows)} held-out rows left from offset {start}")

    description = None
    if args.mode == "shift":
        result = shifted(rows, args.shift_columns.split(","), args.shift_std, args.fraction, args.seed)
        rows, description = result["frame"], result["description"]
        print("injected shift:", json.dumps(description, indent=2), flush=True)

    ready = requests.get(f"{args.url}/health", timeout=10)
    if ready.status_code != 200:
        sys.exit(f"service not ready: {ready.status_code} {ready.text}")

    ids, alerts, version, t0 = [], 0, None, time.perf_counter()
    for i in range(0, len(rows), args.batch):
        chunk = rows.iloc[i : i + args.batch]
        r = requests.post(f"{args.url}/predict", json={"instances": payload(chunk)}, timeout=60)
        r.raise_for_status()
        body = r.json()
        version = body["model_version"]
        ids += [p["request_id"] for p in body["predictions"]]
        alerts += sum(p["alert"] for p in body["predictions"])
    seconds = time.perf_counter() - t0
    print(
        f"sent {len(rows)} rows ({args.mode}) in {seconds:.1f}s to model version {version}; "
        f"alerts {alerts} ({alerts / len(rows):.1%}); image {body['image_tag']}",
        flush=True,
    )
    if args.labels:
        labels = [{"request_id": rid, "y": int(y)} for rid, y in zip(ids, rows["y"], strict=True)]
        for i in range(0, len(labels), 1000):
            requests.post(f"{args.url}/feedback", json={"labels": labels[i : i + 1000]}, timeout=60).raise_for_status()
        print(f"sent {len(labels)} labels", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
