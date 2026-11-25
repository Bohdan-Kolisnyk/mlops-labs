"""Settings, the request log and the notification helper shared by the Lab 4 programs.

Every setting comes from the environment (a ConfigMap or a Secret in the cluster), with the
defaults of a local run. The request log is two JSON-lines files: what the service served
(features, probability, timestamp: the current window of the drift job) and the labels that
arrived later, joined by request id.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"environment variable {name} is not set")
    return value


def model_name() -> str:
    return env("MODEL_NAME", "truth-social-abnormal-move")


def model_alias() -> str:
    return env("MODEL_ALIAS", "champion")


def data_dir() -> Path:
    return Path(env("DATA_DIR", "/data"))


def request_dir() -> Path:
    return Path(env("REQUEST_LOG_DIR", str(data_dir() / "requests")))


def reference_dir() -> Path:
    return Path(env("REFERENCE_DIR", str(data_dir() / "reference")))


def report_dir() -> Path:
    return Path(env("REPORT_DIR", str(data_dir() / "reports")))


def seed_dir() -> Path:
    return Path(env("SEED_DIR", "/app/deploy/seed"))


def utc_now() -> datetime:
    return datetime.now(UTC)


class RequestLog:
    """Append-only JSON lines: requests.jsonl (served rows) and labels.jsonl (late labels)."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.requests_path = self.dir / "requests.jsonl"
        self.labels_path = self.dir / "labels.jsonl"
        self._lock = threading.Lock()

    def _append(self, path: Path, records: list[dict]) -> None:
        lines = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records)
        with self._lock, path.open("a", encoding="utf-8") as f:
            f.write(lines)
            f.flush()

    def append_requests(self, records: list[dict]) -> None:
        self._append(self.requests_path, records)

    def append_labels(self, records: list[dict]) -> None:
        self._append(self.labels_path, records)

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_json(path, lines=True, dtype=False)

    def requests(self) -> pd.DataFrame:
        df = self._read(self.requests_path)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        return df

    def labels(self) -> pd.DataFrame:
        df = self._read(self.labels_path)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        return df


def notify(text: str) -> bool:
    """Send `text` to the Telegram chat of the owner when a bot is configured; never raises."""
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        print(f"[notify] no Telegram bot configured; message was: {text}", flush=True)
        return False
    body = json.dumps({"chat_id": chat, "text": text}).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"[notify] Telegram call failed: {exc}", flush=True)
        return False
