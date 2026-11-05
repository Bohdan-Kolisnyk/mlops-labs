"""Connection settings for the Feast scripts, taken from the same .env as docker compose.

feature_store.yaml refers to ${POSTGRES_USER}, ${POSTGRES_PASSWORD} and ${FEAST_DB_HOST}, and
the S3 client (MinIO) reads AWS_ENDPOINT_URL and the AWS_* keys. On the host they come from
../.env; inside the Airflow container docker compose sets them, and the same function runs.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def setup_env() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    os.environ.setdefault("FEAST_DB_HOST", "localhost")
    endpoint = os.environ.get("FEAST_S3_ENDPOINT") or os.environ.get(
        "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"
    )
    os.environ["AWS_ENDPOINT_URL"] = endpoint
    # MinIO takes the root user as the access key; do not let stray AWS settings win
    if os.environ.get("MINIO_ROOT_USER"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["MINIO_ROOT_USER"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["MINIO_ROOT_PASSWORD"]
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
