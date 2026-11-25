"""Alertmanager webhook that turns a firing drift alert into one retraining Job (stretch goal).

    python -m mlops_labs.serve.responder

POST /alerts takes the Alertmanager webhook payload. Each firing alert is logged. When its name is
in TRIGGER_ALERTS and no retrain Job started in the last COOLDOWN_SECONDS is still around, the
responder creates a Job from the ML image through the Kubernetes API (its ServiceAccount may only
create and list Jobs in its own namespace). Nobody types anything: alert -> webhook -> Job.
Standard library only, so the pod is small.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

API = "https://kubernetes.default.svc"
TOKEN = "/var/run/secrets/kubernetes.io/serviceaccount/token"
CA = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
NAMESPACE_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"


def kube(method: str, path: str, body: dict | None = None) -> dict:
    token = Path(TOKEN).read_text(encoding="utf-8").strip()
    request = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    context = ssl.create_default_context(cafile=CA)
    with urllib.request.urlopen(request, context=context, timeout=20) as response:
        return json.load(response)


def namespace() -> str:
    return Path(NAMESPACE_FILE).read_text(encoding="utf-8").strip()


def recent_retrain_jobs(ns: str, cooldown: int) -> list[str]:
    jobs = kube("GET", f"/apis/batch/v1/namespaces/{ns}/jobs?labelSelector=app%3Dretrain")["items"]
    now, recent = datetime.now(UTC), []
    for job in jobs:
        started = datetime.fromisoformat(job["metadata"]["creationTimestamp"])
        if (now - started).total_seconds() < cooldown:
            recent.append(job["metadata"]["name"])
    return recent


def job_spec(ns: str, alert: str) -> dict:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    image = os.environ["RETRAIN_IMAGE"]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"retrain-{stamp}",
            "namespace": ns,
            "labels": {"app": "retrain"},
            "annotations": {"triggered-by": f"alertmanager:{alert}"},
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "metadata": {"labels": {"app": "retrain"}},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "retrain",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["python", "-m", "mlops_labs.serve.retrain"],
                            "args": ["--trigger", f"alert:{alert}"],
                            "envFrom": [
                                {"configMapRef": {"name": os.environ["APP_CONFIGMAP"]}},
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


def handle_alerts(payload: dict) -> list[str]:
    triggers = {a.strip() for a in os.environ.get("TRIGGER_ALERTS", "").split(",") if a.strip()}
    cooldown = int(os.environ.get("COOLDOWN_SECONDS", "3600"))
    started = []
    for alert in payload.get("alerts", []):
        name = alert.get("labels", {}).get("alertname", "?")
        status = alert.get("status", "?")
        print(f"alert {name} {status}: {alert.get('annotations', {}).get('summary', '')}", flush=True)
        if status != "firing" or name not in triggers or started:
            continue
        ns = namespace()
        recent = recent_retrain_jobs(ns, cooldown)
        if recent:
            print(f"retrain already started within {cooldown}s: {recent}, not starting another", flush=True)
            continue
        spec = job_spec(ns, name)
        kube("POST", f"/apis/batch/v1/namespaces/{ns}/jobs", spec)
        started.append(spec["metadata"]["name"])
        print(f"started Job {spec['metadata']['name']} because of alert {name}", flush=True)
    return started


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._reply(200, {"status": "ok"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            started = handle_alerts(payload)
        except (json.JSONDecodeError, urllib.error.URLError, OSError) as exc:
            print(f"error: {type(exc).__name__}: {exc}", flush=True)
            self._reply(500, {"error": str(exc)})
            return
        self._reply(200, {"started": started})

    def log_message(self, *args):
        return


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    print(f"responder listening on {port}, triggers: {os.environ.get('TRIGGER_ALERTS', '')}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
