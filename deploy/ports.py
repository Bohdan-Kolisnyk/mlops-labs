"""Open the UIs and the services on localhost until Ctrl-C (kubectl port-forward for each).

    python deploy/ports.py

Grafana http://localhost:3000 (admin, password in deploy/.cluster.env; anonymous viewing is on),
Prometheus :9090, Alertmanager :9093, MLflow :5000, Pushgateway :9091, ML service :8000,
LLM service :8001.
"""

from __future__ import annotations

import subprocess
import sys
import time

FORWARDS = [
    ("svc/grafana", 3000, 3000),
    ("svc/prometheus", 9090, 9090),
    ("svc/alertmanager", 9093, 9093),
    ("svc/mlflow", 5000, 5000),
    ("svc/pushgateway", 9091, 9091),
    ("svc/ml-service", 8000, 8000),
    ("svc/llm-service", 8001, 8000),
]


def main() -> int:
    procs = [
        subprocess.Popen(["kubectl", "-n", "mlops", "port-forward", target, f"{local}:{remote}"])
        for target, local, remote in FORWARDS
    ]
    print("forwarding; press Ctrl-C to stop", flush=True)
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
