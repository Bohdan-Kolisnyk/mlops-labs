"""Helpers shared by the deploy scripts: run kubectl and minikube, port-forward, call the services.

Standard library only, so the scripts run with any Python 3.10+ (`python deploy/up.py`).
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NS = "mlops"
IMAGES = {
    "mlops-labs-ml": "deploy/docker/Dockerfile.ml",
    "mlops-labs-llm": "deploy/docker/Dockerfile.llm",
    "mlops-labs-monitor": "deploy/docker/Dockerfile.monitor",
}
MLFLOW_IMAGE = "mlops-labs-mlflow:3.16.1-k8s"


def say(message: str) -> None:
    print(message, flush=True)


def need(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        sys.exit(f"`{tool}` is not on PATH: install it first (README, section Lab 4)")
    return path


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = False,
    input_text: str | None = None,
    timeout: float | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        input=input_text,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=cwd or ROOT,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stdout or "") + (result.stderr or "")
        sys.exit(f"command failed ({result.returncode}): {' '.join(cmd)}\n{detail}")
    return result


def kubectl(*args: str, namespaced: bool = True, **kwargs) -> subprocess.CompletedProcess:
    base = ["kubectl"] + (["-n", NS] if namespaced else [])
    return run([*base, *args], **kwargs)


def git(*args: str) -> str:
    return run(["git", *args], capture=True).stdout.strip()


def image_tag() -> tuple[str, str]:
    """(tag, full commit): the tag is the short commit, plus -dirty when the code has changes."""
    full = git("rev-parse", "HEAD")
    short = git("rev-parse", "--short", "HEAD")
    dirty = git("status", "--porcelain", "--", "src", "rag", "deploy", "scripts", "pyproject.toml")
    return (f"{short}-dirty" if dirty else short), full


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def forward(target: str, remote_port: int, local_port: int | None = None):
    """kubectl port-forward to a service or pod while the block runs; yields the base URL."""
    port = local_port or free_port()
    proc = subprocess.Popen(
        ["kubectl", "-n", NS, "port-forward", target, f"{port}:{remote_port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                sys.exit(f"port-forward to {target} failed: {proc.stdout.read()}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def http(
    method: str, url: str, body: dict | None = None, timeout: float = 60
) -> tuple[int, dict | str]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw, status = response.read().decode("utf-8"), response.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read().decode("utf-8"), exc.code
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def wait_job(name: str, timeout_s: float) -> None:
    """Wait for a Job to complete; stop early, with the pod log, when it has used its retries."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        out = kubectl("get", "job", name, "-o", "json", capture=True, check=False)
        if out.returncode == 0:
            status = json.loads(out.stdout).get("status", {})
            spec = json.loads(out.stdout).get("spec", {})
            if status.get("succeeded", 0) >= 1:
                return
            if status.get("failed", 0) > spec.get("backoffLimit", 0) and not status.get("active"):
                log = kubectl("logs", f"job/{name}", "--tail=40", capture=True, check=False)
                sys.exit(f"job {name} failed:\n{log.stdout}{log.stderr}")
        time.sleep(5)
    sys.exit(f"job {name} did not finish in {timeout_s:.0f}s: kubectl -n {NS} describe job {name}")


def pods_text() -> str:
    return kubectl("get", "pods,svc", capture=True).stdout


def deployment_image(name: str) -> str:
    out = kubectl("get", "deployment", name, "-o", "jsonpath={.spec.template.spec.containers[0].image}",
                  capture=True)
    return out.stdout.strip()


def job_manifest(name: str, image: str, args: list[str], app: str) -> str:
    return json.dumps(
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": name, "namespace": NS, "labels": {"app": app}},
            "spec": {
                "backoffLimit": 0,
                "ttlSecondsAfterFinished": 86400,
                "template": {
                    "metadata": {"labels": {"app": app}},
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": app,
                                "image": image,
                                "imagePullPolicy": "IfNotPresent",
                                "command": ["python", "-m", *args],
                                "envFrom": [
                                    {"configMapRef": {"name": "app-config"}},
                                    {"secretRef": {"name": "notify-secrets", "optional": True}},
                                ],
                                "env": [{"name": "IMAGE_TAG", "value": image.rsplit(":", 1)[-1]}],
                                "resources": {
                                    "requests": {"cpu": "200m", "memory": "512Mi"},
                                    "limits": {"memory": "1Gi"},
                                },
                                "volumeMounts": [{"name": "data", "mountPath": "/data"}],
                            }
                        ],
                        "volumes": [
                            {"name": "data", "persistentVolumeClaim": {"claimName": "mlops-data"}}
                        ],
                    },
                },
            },
        }
    )


def run_job(name: str, image: str, args: list[str], app: str, timeout_s: float = 600) -> str:
    """Create a Job from `python -m <args>` in the given image, wait, return its log."""
    kubectl("apply", "-f", "-", input_text=job_manifest(name, image, args, app))
    wait_job(name, timeout_s)
    return kubectl("logs", f"job/{name}", capture=True).stdout


class Tee:
    """Print and keep the lines, so that a demonstration can be saved as evidence."""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        say(text)
        self.lines.append(text)

    def save(self, path: Path, append: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(self.lines) + "\n"
        if append and path.exists():
            text = path.read_text(encoding="utf-8") + "\n" + text
        path.write_text(text, encoding="utf-8")
        say(f"saved: {path.relative_to(ROOT)}")


def sample_row() -> dict:
    return {
        "text": "Tariffs on steel and aluminum start tomorrow.", "text_len": 44, "link_count": 0,
        "caps_share": 0.05, "has_media": 0, "slot_index": 1, "weekday": 2,
        "minutes_since_open": 90, "sigma": 0.003, "last_bar_return": 0.0, "posts_prev_60min": 0,
    }
