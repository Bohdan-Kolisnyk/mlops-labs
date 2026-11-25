"""The scheduled drift job: reference window against current window, numbers to the Pushgateway.

    python -m mlops_labs.serve.monitor

reference window  deploy/seed training rows (a sample of 1,000), scored by the champion; written by
                  the bootstrap job to REFERENCE_DIR
current window   the last WINDOW_ROWS rows the ML service served (its request log), only rows scored
                  by the model version the reference was scored with
input drift      Evidently DataDriftPreset over the ten input features: Kolmogorov-Smirnov test for
                  numeric columns, chi-squared test for categorical ones, p-value per column and the
                  share of drifted columns (a column drifts below p = 0.05)
prediction drift the same test on the predicted probability, and the alert rate against the
                  reference alert rate
quality          PR-AUC, precision and coverage at the alert threshold on the window rows whose
                  labels have arrived (POST /feedback), against the validation numbers of the model
data freshness   the timestamp of the newest served request, so that an empty or stale window is an
                  alert of its own (a green job over an empty table proves nothing)

The job always pushes the freshness and run metrics, even when the window is too small to test.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from sklearn.metrics import average_precision_score

from mlops_labs.serve import common

NUMERIC = [
    "text_len",
    "link_count",
    "caps_share",
    "sigma",
    "last_bar_return",
    "minutes_since_open",
    "posts_prev_60min",
]
CATEGORICAL = ["has_media", "slot_index", "weekday"]
INPUTS = [*NUMERIC, *CATEGORICAL]
P_THRESHOLD = 0.05
COLUMN_ID = re.compile(r"ValueDrift\(column=([^,)]+)")
MIN_LABELED = 30


def settings() -> dict:
    return {
        "window_rows": int(common.env("WINDOW_ROWS", "300")),
        "min_rows": int(common.env("MIN_WINDOW_ROWS", "100")),
        "share_threshold": float(common.env("DRIFT_SHARE_THRESHOLD", "0.3")),
        "alert_p": float(common.env("DRIFT_ALERT_PVALUE", "0.001")),
        "pushgateway": common.env("PUSHGATEWAY_URL", ""),
        "job": common.env("MONITOR_JOB_NAME", "drift_monitor"),
    }


def _prepare(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    out = df[[*INPUTS, *(extra or [])]].copy()
    out["last_bar_return"] = out["last_bar_return"].fillna(0.0)
    for col in NUMERIC + (extra or []):
        out[col] = out[col].astype("float64")
    for col in CATEGORICAL:
        out[col] = out[col].astype("int64").astype("category")
    return out


def run_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], share: float,
               html: Path | None, categorical: list[str]) -> tuple[dict, float]:
    """Evidently DataDriftPreset on `columns`; returns ({column: p-value}, share of drifted)."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    numerical = [c for c in columns if c not in categorical]
    definition = DataDefinition(numerical_columns=numerical, categorical_columns=categorical)
    ref = Dataset.from_pandas(reference[columns], data_definition=definition)
    cur = Dataset.from_pandas(current[columns], data_definition=definition)
    preset = DataDriftPreset(
        columns=columns,
        drift_share=share,
        num_method="ks",
        cat_method="chisquare",
        threshold=P_THRESHOLD,
    )
    snapshot = Report([preset]).run(cur, ref)
    if html is not None:
        html.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html))
    pvalues, share_value = {}, float("nan")
    for metric in snapshot.dict()["metrics"]:
        metric_id = metric["metric_id"]
        found = COLUMN_ID.match(metric_id)
        if found:
            pvalues[found.group(1)] = float(metric["value"])
        elif metric_id.startswith("DriftedColumnsCount"):
            share_value = float(metric["value"]["share"])
    return pvalues, share_value


def quality(window: pd.DataFrame, labels: pd.DataFrame, meta: dict) -> dict:
    out = {"labeled_rows": 0.0}
    if labels.empty or window.empty:
        return out
    lab = labels.sort_values("ts").drop_duplicates("request_id", keep="last")
    joined = window.merge(lab[["request_id", "y"]], on="request_id", how="inner")
    out["labeled_rows"] = float(len(joined))
    if len(joined) < MIN_LABELED:
        return out
    y = joined["y"].to_numpy().astype(int)
    out["base_rate"] = float(y.mean())
    flagged = joined["alert"].to_numpy().astype(bool)
    if y.min() != y.max():
        out["pr_auc"] = float(average_precision_score(y, joined["probability"].to_numpy()))
    out["alert_rate_labeled"] = float(flagged.mean())
    if flagged.any():
        out["precision_at_alert"] = float(y[flagged].mean())
    if y.sum() > 0:
        out["coverage_at_alert"] = float(y[flagged].sum() / y.sum())
    out["reference_pr_auc"] = float(meta.get("val_pr_auc", np.nan))
    out["reference_base_rate"] = float(meta.get("val_base_rate", np.nan))
    return out


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.values: dict[str, float] = {}

    def set(self, name: str, value: float, doc: str = "", **labels: str) -> None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return
        gauge = self.registry._names_to_collectors.get(name)
        if gauge is None:
            gauge = Gauge(name, doc or name, list(labels), registry=self.registry)
        (gauge.labels(**labels) if labels else gauge).set(float(value))
        self.values[name + (str(sorted(labels.items())) if labels else "")] = float(value)

    def push(self, url: str, job: str) -> None:
        if url:
            push_to_gateway(url, job=job, registry=self.registry)


def main() -> int:
    cfg = settings()
    now = common.utc_now()
    started = time.perf_counter()
    m = Metrics()
    m.set("drift_run_success", 0, "1 when the last run finished")
    try:
        exit_code = run(cfg, m, now)
    except Exception:
        m.set("drift_run_success", 0)
        m.set("drift_run_timestamp_seconds", now.timestamp(), "start of the last run")
        m.push(cfg["pushgateway"], cfg["job"])
        raise
    m.set("drift_run_duration_seconds", time.perf_counter() - started, "duration of the last run")
    m.push(cfg["pushgateway"], cfg["job"])
    return exit_code


def run(cfg: dict, m: Metrics, now) -> int:
    ref_dir = common.reference_dir()
    reference = pd.read_parquet(ref_dir / "reference.parquet")
    meta = json.loads((ref_dir / "reference_meta.json").read_text(encoding="utf-8"))
    log = common.RequestLog(common.request_dir())
    requests, labels = log.requests(), log.labels()

    newest = requests["ts"].max().timestamp() if not requests.empty else 0.0
    m.set("drift_run_success", 1)
    m.set("drift_run_timestamp_seconds", now.timestamp(), "start of the last run")
    m.set("drift_newest_request_timestamp_seconds", newest, "newest request in the log")
    m.set("drift_logged_rows_total", float(len(requests)), "rows in the request log")
    m.set("drift_reference_rows", float(len(reference)), "rows in the reference window")
    m.set("drift_reference_model_version", float(meta["model_version"]), "model of the reference")

    window = pd.DataFrame()
    if not requests.empty:
        same = requests[requests["model_version"].astype(str) == str(meta["model_version"])]
        window = same.sort_values("ts").tail(cfg["window_rows"]).reset_index(drop=True)
    m.set("drift_window_rows", float(len(window)), "rows in the current window")
    m.set("drift_window_sufficient", float(len(window) >= cfg["min_rows"]), "window large enough")
    print(
        f"window: {len(window)} rows (minimum {cfg['min_rows']}), reference: {len(reference)} rows "
        f"of model version {meta['model_version']}, log: {len(requests)} rows, "
        f"newest request: {pd.Timestamp(newest, unit='s', tz='UTC') if newest else 'none'}",
        flush=True,
    )
    if len(window) < cfg["min_rows"]:
        print("window too small to test: only freshness and run metrics are pushed", flush=True)
        return 0

    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    reports = common.report_dir()
    ref_in = _prepare(reference)
    cur_in = _prepare(window)
    pvalues, share = run_report(ref_in, cur_in, INPUTS, cfg["share_threshold"],
                                reports / f"input_drift_{stamp}.html", CATEGORICAL)
    drifted = {c: p < P_THRESHOLD for c, p in pvalues.items()}
    for column, p in pvalues.items():
        m.set("drift_feature_pvalue", p, "KS or chi-squared p-value of the feature", feature=column)
        m.set("drift_feature_detected", float(drifted[column]), "1 when p < 0.05", feature=column)
    m.set("drift_share_of_drifted_columns", share, "share of input columns with drift")
    m.set("drift_columns_drifted", float(sum(drifted.values())), "input columns with drift")
    m.set("drift_share_threshold", cfg["share_threshold"], "alert threshold of the share")
    m.set("drift_pvalue_alert_threshold", cfg["alert_p"], "alert threshold of a p-value")

    ref_pred = _prepare(reference, ["probability"])
    cur_pred = _prepare(window, ["probability"])
    p_pred, _ = run_report(ref_pred, cur_pred, ["probability"], 1.0,
                           reports / f"prediction_drift_{stamp}.html", [])
    m.set("drift_prediction_pvalue", p_pred["probability"], "KS p-value of the probability")
    m.set("drift_prediction_detected", float(p_pred["probability"] < P_THRESHOLD), "1 when p < 0.05")
    m.set("drift_prediction_mean", float(window["probability"].mean()), "mean probability, window")
    m.set("drift_prediction_reference_mean", float(reference["probability"].mean()), "mean, reference")
    m.set("drift_alert_rate", float(window["alert"].astype(bool).mean()), "alert rate, window")
    m.set("drift_alert_rate_reference", float(meta["alert_rate"]), "alert rate, reference")

    q = quality(window, labels, meta)
    for key, value in q.items():
        m.set(f"quality_{key}", value, f"quality metric {key} on the labeled window rows")

    summary = {
        "at": now.isoformat(),
        "window_rows": len(window),
        "share_of_drifted_columns": share,
        "share_threshold": cfg["share_threshold"],
        "feature_pvalues": pvalues,
        "prediction_pvalue": p_pred["probability"],
        "alert_rate": float(window["alert"].astype(bool).mean()),
        "quality": q,
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"summary_{stamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{'feature':22} {'p-value':>10}  drift", flush=True)
    for column, p in sorted(pvalues.items(), key=lambda kv: kv[1]):
        print(f"{column:22} {p:10.4g}  {'DRIFT' if drifted[column] else '-'}", flush=True)
    print(
        f"share of drifted columns {share:.3f} (alert threshold {cfg['share_threshold']}), "
        f"prediction p-value {p_pred['probability']:.4g}, alert rate {summary['alert_rate']:.3f} "
        f"(reference {meta['alert_rate']:.3f}), labeled rows {int(q['labeled_rows'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
