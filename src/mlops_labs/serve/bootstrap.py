"""Fill the in-cluster registry: train the Lab 2 candidate on the committed seed and register it.

    python -m mlops_labs.serve.bootstrap --if-missing           # first start: v1 as `champion`
    python -m mlops_labs.serve.bootstrap --c 0.3 --alias none   # another version, alias untouched
    python -m mlops_labs.serve.bootstrap --refresh-reference    # reference window for the champion

The seed (deploy/seed/train_table.parquet) is the Lab 2 training table, so a clean clone needs
neither DVC nor MinIO. The model is the same TF-IDF and logistic regression pipeline, trained by
the same code on the same train split. Next to the model the job writes the drift reference:
a sample of the training rows with the probability the champion gives them. The operating point
(threshold of the top 20% of the validation scores) is stored as a tag of the model version, so
the service reads it from the registry together with the model.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from mlops_labs import models
from mlops_labs.features import FEATURE_COLUMNS
from mlops_labs.serve import common

EXPERIMENT = "truth-social-abnormal-moves"
REFERENCE_ROWS = 1000


def wait_for_registry(timeout_s: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            MlflowClient().search_experiments(max_results=1)
            return
        except Exception as exc:
            if time.monotonic() > deadline:
                raise RuntimeError(f"the tracking server did not answer: {exc}") from exc
            print(f"waiting for the tracking server: {type(exc).__name__}", flush=True)
            time.sleep(3)


def load_seed(seed_dir: Path) -> tuple[pd.DataFrame, str]:
    table = pd.read_parquet(seed_dir / "train_table.parquet")
    version = (seed_dir / "DATA_VERSION").read_text(encoding="utf-8").strip()
    return table, version


def alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None


def train_and_register(
    table: pd.DataFrame,
    data_version: str,
    c: float,
    seed: int,
    alias: str | None,
    extra_tags: dict[str, str] | None = None,
    experiment: str = EXPERIMENT,
    extra_train: pd.DataFrame | None = None,
) -> dict:
    """Fit on the train split (plus `extra_train`, labeled rows served later), register a new
    version with lineage and its operating point. Validation and test stay the seed splits."""
    parts = {s: table[table["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    fit_rows = parts["train"]
    if extra_train is not None:
        columns = ["text", *FEATURE_COLUMNS, "y"]
        fit_rows = pd.concat([fit_rows[columns], extra_train[columns]], ignore_index=True)
    model = models.Candidate(c, seed).fit(fit_rows)
    val_scores, test_scores = model.score(parts["val"]), model.score(parts["test"])
    threshold = models.alert_threshold(val_scores, seed)
    metrics = {}
    for split, scores in (("val", val_scores), ("test", test_scores)):
        y = parts[split]["y"].to_numpy()
        point = models.point_metrics(y, scores, True)
        op = models.operating_point(y, scores, threshold, seed + 1)
        metrics.update({f"{split}_{k}": v for k, v in point.items()})
        metrics[f"{split}_coverage_at_20"] = op["coverage"]
        metrics[f"{split}_precision_at_20"] = op["precision"]

    commit = os.environ.get("GIT_COMMIT", "unknown")
    name = common.model_name()
    tags = {
        "data_version": data_version,
        "git_commit": commit,
        "alert_threshold": f"{threshold:.8f}",
        "val_pr_auc": f"{metrics['val_pr_auc']:.6f}",
        "test_pr_auc": f"{metrics['test_pr_auc']:.6f}",
        "val_base_rate": f"{metrics['val_base_rate']:.6f}",
        "source": "seed",
        "C": str(c),
        **(extra_tags or {}),
    }
    mlflow.set_experiment(experiment)
    x_train = models.prepare_candidate_input(fit_rows)
    with mlflow.start_run(run_name=f"{model.name}-seed-C{c}") as run:
        mlflow.log_params(
            {
                "model": model.name,
                "C": c,
                "seed": seed,
                "features": ",".join(FEATURE_COLUMNS),
                "n_train": len(parts["train"]),
                "n_extra_train": 0 if extra_train is None else len(extra_train),
                "n_val": len(parts["val"]),
                "n_test": len(parts["test"]),
                "sklearn_version": sklearn.__version__,
                "data_source": "seed",
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"data_version": data_version, "git_commit": commit, "role": "candidate"})
        signature = infer_signature(x_train, model.pipeline_.predict_proba(x_train)[:, 1])
        info = mlflow.sklearn.log_model(
            model.pipeline_,
            name="model",
            signature=signature,
            input_example=x_train.head(3),
            pyfunc_predict_fn="predict_proba",
            pip_requirements=[
                f"scikit-learn=={sklearn.__version__}",
                f"pandas=={pd.__version__}",
                f"numpy=={np.__version__}",
                "skops",
            ],
        )
        tags["source_run_id"] = run.info.run_id
        mv = mlflow.register_model(info.model_uri, name, tags=tags)
    client = MlflowClient()
    if alias:
        client.set_registered_model_alias(name, alias, mv.version)
    print(
        f"registered {name} version {mv.version}"
        + (f", alias {alias!r}" if alias else ", no alias")
        + f", val PR-AUC {metrics['val_pr_auc']:.4f}, threshold {threshold:.4f}",
        flush=True,
    )
    return {
        "version": str(mv.version),
        "metrics": metrics,
        "threshold": threshold,
        "pipeline": model.pipeline_,
    }


def write_reference(table: pd.DataFrame, directory: Path, alias: str) -> dict:
    """Reference window: a sample of the training rows, scored by the model behind `alias`."""
    name = common.model_name()
    mv = MlflowClient().get_model_version_by_alias(name, alias)
    model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    train = table[table["split"] == "train"]
    sample = train.sample(n=min(REFERENCE_ROWS, len(train)), random_state=0).reset_index(drop=True)
    proba = model.predict_proba(models.prepare_candidate_input(sample))[:, 1]
    threshold = float(mv.tags["alert_threshold"])
    ref = sample[["text", *FEATURE_COLUMNS]].copy()
    ref["probability"] = proba
    ref["alert"] = proba >= threshold
    directory.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkstemp(dir=directory, suffix=".parquet")[1])
    ref.to_parquet(tmp, index=False)
    tmp.replace(directory / "reference.parquet")
    meta = {
        "model_name": name,
        "model_version": str(mv.version),
        "alias": alias,
        "data_version": mv.tags.get("data_version", "?"),
        "rows": len(ref),
        "alert_threshold": threshold,
        "alert_rate": float(ref["alert"].mean()),
        "val_pr_auc": float(mv.tags.get("val_pr_auc", "nan")),
        "val_base_rate": float(mv.tags.get("val_base_rate", "nan")),
        "written_at": common.utc_now().isoformat(),
    }
    (directory / "reference_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"reference window: {meta['rows']} rows scored by version {meta['model_version']}", flush=True)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alias", default="champion", help="alias to point at the new version, or 'none'")
    ap.add_argument("--if-missing", action="store_true", help="do nothing when the alias exists")
    ap.add_argument("--refresh-reference", action="store_true", help="only rewrite the reference")
    ap.add_argument("--seed-dir", default=None)
    args = ap.parse_args(argv)

    seed_dir = Path(args.seed_dir) if args.seed_dir else common.seed_dir()
    table, data_version = load_seed(seed_dir)
    wait_for_registry()
    client = MlflowClient()
    name, champion_alias = common.model_name(), common.model_alias()
    ref_dir = common.reference_dir()

    if args.refresh_reference:
        write_reference(table, ref_dir, champion_alias)
        return 0
    alias = None if args.alias == "none" else args.alias
    if args.if_missing and alias_version(client, name, alias or champion_alias):
        print(f"{name}@{alias or champion_alias} already exists: nothing to train", flush=True)
        if not (ref_dir / "reference.parquet").exists():
            write_reference(table, ref_dir, champion_alias)
        return 0
    train_and_register(table, data_version, args.c, args.seed, alias)
    if alias == champion_alias:
        write_reference(table, ref_dir, champion_alias)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
