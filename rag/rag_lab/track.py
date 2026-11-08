"""MLflow logging: the same server as Lab 2."""

from __future__ import annotations

import os
from pathlib import Path


def setup(cfg: dict) -> None:
    import mlflow

    from rag_lab.config import load_env

    load_env()
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri_default"])
    )


def log_run(
    experiment: str,
    run_name: str,
    params: dict,
    metrics: dict,
    files: list[Path] | None = None,
    tags: dict | None = None,
) -> str:
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items()})
        for f in files or []:
            mlflow.log_artifact(str(f))
        return run.info.run_id
