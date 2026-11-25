"""The readiness test of lab 4: start the ML service with its model absent, and keep the output.

    python deploy/missing_model.py

The alias in the ConfigMap is pointed at a version that does not exist and the Deployment is scaled
0 -> 1. The pod must stay 0/1, the Service must have no endpoints (no traffic), /health must answer
503 with the reason. Then the alias is restored and the service becomes ready again.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kube
from kube import ROOT, Tee, kubectl


def patch_alias(alias: str) -> None:
    patch = json.dumps({"data": {"MODEL_ALIAS": alias}})
    kubectl("patch", "configmap", "app-config", "--type=merge", "-p", patch)


def main() -> int:
    out = Tee()
    out("== 1. the model is absent: alias 'no-such-alias' ==")
    kubectl("scale", "deployment/ml-service", "--replicas=0")
    kubectl("wait", "--for=delete", "pod", "-l", "app=ml-service", "--timeout=120s", check=False)
    patch_alias("no-such-alias")
    kubectl("scale", "deployment/ml-service", "--replicas=1")
    time.sleep(35)
    out("$ kubectl get pods -l app=ml-service")
    out(kubectl("get", "pods", "-l", "app=ml-service", capture=True).stdout.rstrip())
    out("$ kubectl get endpoints ml-service")
    out(kubectl("get", "endpoints", "ml-service", capture=True).stdout.rstrip())
    pod = kubectl("get", "pods", "-l", "app=ml-service", "-o", "jsonpath={.items[0].metadata.name}",
                  capture=True).stdout.strip()
    with kube.forward(f"pod/{pod}", 8000) as url:
        status, body = kube.http("GET", f"{url}/health")
        out(f"GET /health on the pod -> {status} {body}")
        status, body = kube.http("GET", f"{url}/livez")
        out(f"GET /livez on the pod  -> {status} {body}")
        status, body = kube.http("POST", f"{url}/predict", {"instances": [kube.sample_row()]})
        out(f"POST /predict on the pod -> {status} {body}")
    out("$ kubectl describe pod (readiness events)")
    events = kubectl("describe", "pod", pod, capture=True).stdout
    out("\n".join(ln for ln in events.splitlines() if "Readiness" in ln or "Ready:" in ln)[:600])
    out("$ kubectl get pods -l app=ml-service   (restarts must stay 0: a missing model is not a liveness failure)")
    out(kubectl("get", "pods", "-l", "app=ml-service", capture=True).stdout.rstrip())

    out("")
    out("== 2. the model is back: alias 'champion' ==")
    patch_alias("champion")
    kubectl("rollout", "restart", "deployment/ml-service")
    kubectl("rollout", "status", "deployment/ml-service", "--timeout=300s")
    out(kubectl("get", "pods", "-l", "app=ml-service", capture=True).stdout.rstrip())
    out(kubectl("get", "endpoints", "ml-service", capture=True).stdout.rstrip())
    with kube.forward("svc/ml-service", 8000) as url:
        status, body = kube.http("GET", f"{url}/health")
        out(f"GET /health -> {status} {body}")
    out.save(ROOT / "docs" / "evidence" / "missing-model.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
