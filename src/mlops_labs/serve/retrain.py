"""The action behind the drift alert: retrain on the seed plus the labeled current window.

    python -m mlops_labs.serve.retrain --trigger alert:FeatureDriftSigma

Started by the responder (a Kubernetes Job created from an Alertmanager webhook), or by hand.
The job reads the rows the ML service served and the labels that came back, trains a candidate
on the seed train split plus the first 70% of the labeled rows (in time order), registers it with
the alias `candidate` and compares it with the champion on two sets the candidate did not train on:
the seed validation split and the last 30% of the labeled window.

RETRAIN_PROMOTE=never   (default) the candidate waits; a person promotes it (design 5.5)
RETRAIN_PROMOTE=guarded the candidate becomes `champion` only if it is not worse than the champion
                        on the seed validation split (tolerance 0.02 PR-AUC) and better on the
                        labeled window holdout; the old champion keeps the alias `previous`, so
                        a rollback is one alias move; the service reloads and the reference is
                        rewritten for the new model
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.request

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from sklearn.metrics import average_precision_score

from mlops_labs import models
from mlops_labs.serve import bootstrap, common

VAL_TOLERANCE = 0.02
HOLDOUT_SHARE = 0.3


def labeled_window(log: common.RequestLog) -> pd.DataFrame:
    requests, labels = log.requests(), log.labels()
    if requests.empty or labels.empty:
        return pd.DataFrame()
    lab = labels.sort_values("ts").drop_duplicates("request_id", keep="last")
    joined = requests.merge(lab[["request_id", "y"]], on="request_id", how="inner")
    return joined.sort_values("ts").reset_index(drop=True)


def pr_auc(model, df: pd.DataFrame) -> float:
    if df["y"].nunique() < 2:
        return float("nan")
    scores = model.predict_proba(models.prepare_candidate_input(df))[:, 1]
    return float(average_precision_score(df["y"], scores))


def push(job_metrics: dict[str, float]) -> None:
    url = common.env("PUSHGATEWAY_URL", "")
    if not url:
        return
    registry = CollectorRegistry()
    for name, value in job_metrics.items():
        if not math.isnan(value):
            Gauge(name, name, registry=registry).set(value)
    push_to_gateway(url, job="retrain", registry=registry)


def reload_service() -> None:
    url = common.env("ML_SERVICE_URL", "")
    if not url:
        return
    request = urllib.request.Request(f"{url}/admin/reload", data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            print(f"service reloaded: {response.read().decode()[:200]}", flush=True)
    except urllib.error.URLError as exc:
        print(f"service reload failed: {exc}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--trigger", default="manual")
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--min-labeled", type=int, default=int(common.env("RETRAIN_MIN_LABELED", "100")))
    args = ap.parse_args(argv)
    promote = common.env("RETRAIN_PROMOTE", "never")

    bootstrap.wait_for_registry()
    table, data_version = bootstrap.load_seed(common.seed_dir())
    window = labeled_window(common.RequestLog(common.request_dir()))
    print(f"trigger {args.trigger}: {len(window)} labeled rows in the request log", flush=True)
    if len(window) < args.min_labeled:
        text = (
            f"Retrain skipped ({args.trigger}): {len(window)} labeled rows, "
            f"{args.min_labeled} needed."
        )
        print(text, flush=True)
        push({"retrain_run_timestamp_seconds": common.utc_now().timestamp(), "retrain_skipped": 1})
        common.notify(text)
        return 0

    cut = int(len(window) * (1 - HOLDOUT_SHARE))
    fit_rows, holdout = window.iloc[:cut], window.iloc[cut:]
    name, client = common.model_name(), MlflowClient()
    champion_mv = client.get_model_version_by_alias(name, common.model_alias())
    champion = mlflow.sklearn.load_model(f"models:/{name}@{common.model_alias()}")

    result = bootstrap.train_and_register(
        table,
        data_version,
        args.c,
        0,
        "candidate",
        extra_tags={
            "source": "retrain",
            "trigger": args.trigger,
            "parent_champion_version": str(champion_mv.version),
            "window_fit_rows": str(len(fit_rows)),
            "window_holdout_rows": str(len(holdout)),
        },
        extra_train=fit_rows,
    )
    candidate = result["pipeline"]
    val = table[table["split"] == "val"].reset_index(drop=True)
    numbers = {
        "candidate_val": pr_auc(candidate, val),
        "champion_val": pr_auc(champion, val),
        "candidate_holdout": pr_auc(candidate, holdout),
        "champion_holdout": pr_auc(champion, holdout),
    }
    ok_val = numbers["candidate_val"] >= numbers["champion_val"] - VAL_TOLERANCE
    ok_hold = numbers["candidate_holdout"] > numbers["champion_holdout"]
    promoted = promote == "guarded" and ok_val and ok_hold
    version = result["version"]
    client.set_model_version_tag(name, version, "decision", "promoted" if promoted else "waiting")
    for key, value in numbers.items():
        client.set_model_version_tag(name, version, f"pr_auc_{key}", f"{value:.6f}")
    if promoted:
        client.set_registered_model_alias(name, "previous", champion_mv.version)
        client.set_registered_model_alias(name, common.model_alias(), version)
        reload_service()
        bootstrap.write_reference(table, common.reference_dir(), common.model_alias())

    text = (
        f"Retrain finished ({args.trigger}). Candidate version {version} vs champion "
        f"{champion_mv.version}: PR-AUC on validation {numbers['candidate_val']:.3f} vs "
        f"{numbers['champion_val']:.3f}, on the labeled window holdout "
        f"{numbers['candidate_holdout']:.3f} vs {numbers['champion_holdout']:.3f}. "
        + (
            "Promoted by the guard; the previous champion keeps the alias 'previous'."
            if promoted
            else f"Not promoted (RETRAIN_PROMOTE={promote}): a person decides. "
            f"Promote with: python deploy/promote.py --version {version}"
        )
    )
    print(text, flush=True)
    print(json.dumps({"numbers": numbers, "promoted": promoted, "version": version}), flush=True)
    push(
        {
            "retrain_run_timestamp_seconds": common.utc_now().timestamp(),
            "retrain_skipped": 0,
            "retrain_candidate_version": float(version),
            "retrain_promoted": float(promoted),
            "retrain_candidate_pr_auc_holdout": numbers["candidate_holdout"],
            "retrain_champion_pr_auc_holdout": numbers["champion_holdout"],
        }
    )
    common.notify(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
