"""Quality gate (design 6.6, data quality row). Any failed check stops the run."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlops_labs import config

REQUIRED_COLUMNS = [
    "post_id",
    "created_at",
    "text",
    "has_media",
    "bar_start",
    "session_date",
    "slot",
    "r",
    "sigma",
    "y",
    "text_empty",
    "dup_text",
]


class QualityGateError(RuntimeError):
    """At least one quality check failed; the batch must not reach the feature table."""


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def run_checks(
    clean: pd.DataFrame,
    bars: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[CheckResult]:
    """Run every check on the cleaned rows of one batch [start, end)."""
    results: list[CheckResult] = []

    def add(name: str, passed: bool, detail: str) -> None:
        results.append(CheckResult(name, bool(passed), detail))

    missing = [c for c in REQUIRED_COLUMNS if c not in clean.columns]
    add("schema", not missing, f"missing columns: {missing}" if missing else "all columns present")
    if missing:
        return results  # the other checks need these columns

    n = len(clean)
    add("row_count", n >= config.MIN_BATCH_ROWS, f"{n} rows, minimum {config.MIN_BATCH_ROWS}")

    nulls = clean[["post_id", "created_at", "r", "sigma", "y"]].isna().sum()
    bad_cols = {k: int(v) for k, v in nulls.items() if v}
    add("no_nulls", not bad_cols, f"nulls: {bad_cols}" if bad_cols else "no nulls in key columns")

    dup_ids = int(clean["post_id"].duplicated().sum())
    add("unique_post_id", dup_ids == 0, f"{dup_ids} duplicated post_id")

    t = pd.to_datetime(clean["created_at"], utc=True)
    outside = int(((t < start) | (t >= end)).sum())
    add("timestamps_in_interval", outside == 0, f"{outside} rows outside [{start}, {end})")

    bad_slot = int((~clean["slot"].isin(config.POST_SLOTS)).sum())
    add("slot_in_window", bad_slot == 0, f"{bad_slot} rows outside the 09:30-14:30 ET window")

    share_empty = float(clean["text_empty"].mean()) if n else 1.0
    add(
        "empty_text_share",
        share_empty <= config.EMPTY_TEXT_MAX_SHARE,
        f"{share_empty:.1%} empty texts, maximum {config.EMPTY_TEXT_MAX_SHARE:.0%}",
    )

    label_rate = float(clean["y"].mean()) if n else float("nan")
    add(
        "label_rate",
        0.0 < label_rate < 0.5,
        f"abnormal share {label_rate:.1%}, expected between 0% and 50%",
    )

    days = bars[
        (bars["session_date"] >= start.tz_localize(None).normalize())
        & (bars["session_date"] < end.tz_localize(None).normalize())
    ]
    per_day = days.groupby("session_date").size()
    allowed = {config.BARS_PER_FULL_DAY, config.BARS_PER_HALF_DAY}
    odd = per_day[~per_day.isin(allowed)]
    add(
        "bars_per_day",
        odd.empty,
        f"{len(odd)} days with a wrong number of bars: {odd.to_dict()}"
        if not odd.empty
        else f"{len(per_day)} trading days, 7 bars (3 on half-days)",
    )
    return results


def assert_quality(
    clean: pd.DataFrame, bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> list[CheckResult]:
    """Run the checks and raise QualityGateError listing every failure."""
    results = run_checks(clean, bars, start, end)
    failed = [r for r in results if not r.passed]
    if failed:
        lines = "; ".join(f"{r.name}: {r.detail}" for r in failed)
        raise QualityGateError(f"quality gate failed ({len(failed)} checks): {lines}")
    return results
