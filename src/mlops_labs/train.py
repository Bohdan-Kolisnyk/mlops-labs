"""Train B0-B2 and the candidate on the time split, track every run in MLflow, and register
the candidate with lineage.

    uv run python -m mlops_labs.train --register          # read the warehouse, register
    uv run python -m mlops_labs.train --source raw        # read data/raw, no registration

Every run carries two lineage tags: `data_version` (DVC hash of the raw data the rows came
from) and `git_commit`. The registered model version carries the same two tags.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score
from sqlalchemy.exc import OperationalError

from mlops_labs import config, dataset, lineage, models
from mlops_labs.data import read_data_version
from mlops_labs.env import ROOT, load_env
from mlops_labs.features import FEATURE_COLUMNS

EXPERIMENT = "truth-social-abnormal-moves"
MODEL_NAME = "truth-social-abnormal-move"
ALIAS = "champion"
CV_MIN_TRAIN_MONTHS = 6
CHECKED_METRICS = ("val_pr_auc", "test_pr_auc")  # what the reproduction command compares


class LineageError(RuntimeError):
    """Registering a model whose data version or code commit cannot be stated."""


def expanding_window_cv(train: pd.DataFrame, c: float, seed: int) -> list[float]:
    """PR-AUC of the candidate on each month of the training period, trained on earlier
    months only (design 5.4)."""
    month = train["session_date"].dt.to_period("M")
    ordered = sorted(month.unique())
    scores = []
    for m in ordered[CV_MIN_TRAIN_MONTHS:]:
        fit, test = train[month < m], train[month == m]
        if test["y"].sum() == 0 or fit["y"].sum() == 0:
            continue
        model = models.Candidate(c, seed).fit(fit)
        scores.append(float(average_precision_score(test["y"], model.score(test))))
    return scores


def _calibration_csv(y: np.ndarray, p: np.ndarray, path: Path) -> None:
    frac, mean_p = calibration_curve(y, np.clip(p, 0, 1), n_bins=5, strategy="quantile")
    pd.DataFrame({"mean_predicted": mean_p, "observed_rate": frac}).to_csv(path, index=False)


def evaluate(table: pd.DataFrame, c: float, seed: int, bootstrap_n: int) -> tuple[dict, dict, dict]:
    """Fit all four models on train; return (fitted models, metrics per model, extras)."""
    split = dataset.split_name(table["session_date"])
    parts = {s: table[split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    for s, p in parts.items():
        if p.empty or p["y"].sum() == 0:
            raise RuntimeError(f"split {s!r} has no rows or no positive examples")

    fitted = {
        m.name: m.fit(parts["train"])
        for m in (
            models.ConstantPrior(),
            models.KeywordRule(),
            models.ContextOnly(),
            models.Candidate(c, seed),
        )
    }
    probabilistic = {"B0-prior", "B2-context", "candidate-tfidf-lr"}
    scores = {s: {n: m.score(parts[s]) for n, m in fitted.items()} for s in ("val", "test")}

    metrics: dict[str, dict[str, float]] = {n: {} for n in fitted}
    for n in fitted:
        threshold = models.alert_threshold(scores["val"][n], seed)
        for s in ("val", "test"):
            y = parts[s]["y"].to_numpy()
            point = models.point_metrics(y, scores[s][n], n in probabilistic)
            op = models.operating_point(y, scores[s][n], threshold, seed + 1)
            for k, v in point.items():
                metrics[n][f"{s}_{k}"] = v
            metrics[n][f"{s}_coverage_at_20"] = op["coverage"]
            metrics[n][f"{s}_precision_at_20"] = op["precision"]
            metrics[n][f"{s}_alert_rate"] = op["alert_rate"]

    best_baseline = max(("B1-keywords", "B2-context"), key=lambda n: metrics[n]["val_pr_auc"])
    for s in ("val", "test"):
        y = parts[s]["y"].to_numpy()
        ci = models.clustered_bootstrap_ci(
            y,
            scores[s],
            parts[s]["bar_start"].to_numpy(),
            bootstrap_n,
            seed,
            reference=best_baseline,
        )
        for n in fitted:
            metrics[n][f"{s}_pr_auc_ci_low"], metrics[n][f"{s}_pr_auc_ci_high"] = ci[n]
        key = f"candidate-tfidf-lr-{best_baseline}"
        cand = metrics["candidate-tfidf-lr"]
        cand[f"{s}_pr_auc_diff_vs_best_baseline"] = (
            cand[f"{s}_pr_auc"] - metrics[best_baseline][f"{s}_pr_auc"]
        )
        cand[f"{s}_pr_auc_diff_ci_low"], cand[f"{s}_pr_auc_diff_ci_high"] = ci[key]

    cv = expanding_window_cv(parts["train"], c, seed)
    metrics["candidate-tfidf-lr"]["cv_pr_auc_mean"] = float(np.mean(cv))
    metrics["candidate-tfidf-lr"]["cv_folds"] = float(len(cv))
    extras = {"parts": parts, "scores": scores, "best_baseline": best_baseline}
    return fitted, metrics, extras


def _check_lineage(data_version: str, commit: str, dirty: bool, allow_dirty: bool) -> None:
    if data_version.startswith("unversioned"):
        raise LineageError(f"data version is {data_version!r}: data/raw.dvc is missing")
    if commit == "unknown":
        raise LineageError("not a git checkout: the code commit is unknown")
    if dirty and not allow_dirty:
        raise LineageError(
            "uncommitted changes in src/, scripts/, pyproject.toml or uv.lock: commit them "
            "first, otherwise the git_commit tag would not describe the code that ran "
            "(or pass --allow-dirty and the run is tagged git_dirty=true)"
        )


def run_training(
    source: str = "warehouse",
    raw_dir: str | Path | None = None,
    c: float = 1.0,
    seed: int = 0,
    bootstrap_n: int = 1000,
    register: bool = False,
    allow_dirty: bool = False,
    extra_tags: dict[str, str] | None = None,
) -> dict:
    load_env()
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    if source == "raw":
        table, data_version = dataset.from_raw(raw_dir or ROOT / "data" / "raw")
    elif source == "warehouse":
        from mlops_labs.warehouse import get_engine

        try:
            table, data_version = dataset.from_warehouse(get_engine())
        except OperationalError as e:
            raise RuntimeError(
                "cannot read the warehouse: is the stack up (docker compose up -d) and does "
                "WAREHOUSE_URL in .env use the same password as POSTGRES_PASSWORD?"
            ) from e
    else:
        raise ValueError(f"unknown source {source!r}")

    commit, dirty = lineage.git_commit(), lineage.git_dirty()
    if register:
        _check_lineage(data_version, commit, dirty, allow_dirty)
        if source == "warehouse":
            pointer = read_data_version(raw_dir or ROOT / "data" / "raw")
            if pointer != data_version:
                raise LineageError(
                    f"the warehouse was built from data {data_version}, but data/raw.dvc names "
                    f"{pointer}: run the ETL for all months again, then train"
                )

    fitted, metrics, extras = evaluate(table, c, seed, bootstrap_n)
    parts, scores = extras["parts"], extras["scores"]

    common_params = {
        "data_source": source,
        "seed": seed,
        "top_rate": models.TOP_RATE,
        "bootstrap_n": bootstrap_n,
        "feature_version": config.FEATURE_VERSION,
        "features": ",".join(FEATURE_COLUMNS),
        "label_k": config.K,
        "sigma_days": config.SIGMA_DAYS,
        "train_end": config.TRAIN_END,
        "val_range": f"{config.VAL_START}..{config.VAL_END}",
        "test_start": config.TEST_START,
        "n_train": len(parts["train"]),
        "n_val": len(parts["val"]),
        "n_test": len(parts["test"]),
        "sklearn_version": sklearn.__version__,
    }
    tags = {
        "data_version": data_version,
        "git_commit": commit,
        "git_dirty": str(dirty).lower(),
        "best_baseline": extras["best_baseline"],
        **(extra_tags or {}),
    }

    mlflow.set_experiment(EXPERIMENT)
    run_ids: dict[str, str] = {}
    for name, model in fitted.items():
        specific = {"model": name}
        if name == "B1-keywords":
            specific["keywords_hash"] = models.keywords_hash()
        if name == "candidate-tfidf-lr":
            specific["C"] = c
        role = "candidate" if name.startswith("candidate") else "baseline"
        with mlflow.start_run(run_name=name) as run:
            run_ids[name] = run.info.run_id
            mlflow.log_params({**common_params, **specific})
            mlflow.log_metrics(metrics[name])
            mlflow.set_tags({**tags, "role": role})
            if name in ("B2-context", "candidate-tfidf-lr"):
                with tempfile.TemporaryDirectory() as tmp:
                    for s in ("val", "test"):
                        path = Path(tmp) / f"calibration_{s}.csv"
                        _calibration_csv(parts[s]["y"].to_numpy(), scores[s][name], path)
                        mlflow.log_artifact(str(path), artifact_path="calibration")
            if name == "candidate-tfidf-lr":
                x_train = models.prepare_candidate_input(parts["train"])
                signature = infer_signature(x_train, model.pipeline_.predict_proba(x_train)[:, 1])
                info = mlflow.sklearn.log_model(
                    model.pipeline_,
                    name="model",
                    signature=signature,
                    input_example=x_train.head(3),
                    pyfunc_predict_fn="predict_proba",
                    # explicit, so MLflow does not export the whole project lock file
                    pip_requirements=[
                        f"scikit-learn=={sklearn.__version__}",
                        f"pandas=={pd.__version__}",
                        f"numpy=={np.__version__}",
                        "skops",
                    ],
                )
                if register:
                    client = MlflowClient()
                    cand = metrics[name]
                    beats = cand["val_pr_auc_diff_ci_low"] > 0  # design 5.4 promotion rule
                    mv = mlflow.register_model(
                        info.model_uri,
                        MODEL_NAME,
                        tags={
                            "data_version": data_version,
                            "git_commit": commit,
                            "beats_best_baseline_on_val": str(beats).lower(),
                            "source_run_id": run.info.run_id,
                        },
                    )
                    client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
                    mlflow.set_tag("registered_version", str(mv.version))
                    print(f"registered {MODEL_NAME} version {mv.version}, alias {ALIAS!r}")

    return {
        "run_ids": run_ids,
        "metrics": metrics,
        "data_version": data_version,
        "git_commit": commit,
        "git_dirty": dirty,
        "best_baseline": extras["best_baseline"],
        "sizes": {s: len(p) for s, p in parts.items()},
    }


def format_report(result: dict) -> str:
    m = result["metrics"]
    rows = ["model                 val PR-AUC  test PR-AUC  test cov@20%  test prec@20%"]
    for name, v in m.items():
        rows.append(
            f"{name:20}  {v['val_pr_auc']:9.4f}  {v['test_pr_auc']:10.4f}  "
            f"{v['test_coverage_at_20']:11.3f}  {v['test_precision_at_20']:12.3f}"
        )
    base = m["B0-prior"]
    rows.append(
        f"base rate: val {base['val_base_rate']:.3f}, test {base['test_base_rate']:.3f}; "
        f"rows: {result['sizes']}"
    )
    c = m["candidate-tfidf-lr"]
    rows.append(
        f"candidate minus {result['best_baseline']} (PR-AUC): "
        f"val {c['val_pr_auc_diff_vs_best_baseline']:+.4f} "
        f"[{c['val_pr_auc_diff_ci_low']:+.4f}, {c['val_pr_auc_diff_ci_high']:+.4f}], "
        f"test {c['test_pr_auc_diff_vs_best_baseline']:+.4f} "
        f"[{c['test_pr_auc_diff_ci_low']:+.4f}, {c['test_pr_auc_diff_ci_high']:+.4f}]"
    )
    rows.append(f"data_version={result['data_version']}  git_commit={result['git_commit']}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", choices=["warehouse", "raw"], default="warehouse")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--C", type=float, default=1.0, dest="c")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000, dest="bootstrap_n")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    try:
        result = run_training(
            source=args.source,
            raw_dir=args.raw_dir,
            c=args.c,
            seed=args.seed,
            bootstrap_n=args.bootstrap_n,
            register=args.register,
            allow_dirty=args.allow_dirty,
        )
    except LineageError as e:
        print(f"refusing to register: {e}", file=sys.stderr)
        return 2
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
