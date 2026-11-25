"""The drift demonstration of lab 4, sections 4 to 6, in one command.

    python deploy/demo.py

1. baseline: 300 unchanged rows through /predict, their labels through /feedback, one monitoring run
2. shift: the documented Badgers shift (sigma +3 batch standard deviations in 40% of the rows), 300
   rows through /predict, labels, one monitoring run
3. the alert: Prometheus rule -> Alertmanager -> Telegram (when configured) and the responder, which
   starts one retrain Job (stretch); the script waits for the alert and for the Job, prints both
Every step prints what it did; the output is saved to docs/evidence/drift-demo.txt.
The monitoring runs here are started by hand from the CronJob template so that the demonstration does
not wait for the 5-minute schedule; the CronJob keeps running on its own as well.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kube
from kube import ROOT, Tee, kubectl


def replay(url: str, mode: str, out: Tee, extra: list[str]) -> None:
    uv = shutil.which("uv")
    base = [uv, "run", "--with", "requests", "--with", "badgers", "python"] if uv else [sys.executable]
    cmd = [*base, "scripts/replay.py", "--url", url, "--mode", mode, "--labels", *extra]
    out("$ " + " ".join(cmd[-8:]))
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False)
    for line in (done.stdout + done.stderr).strip().splitlines():
        out("  " + line)
    if done.returncode != 0:
        sys.exit("replay failed")


def monitor_run(label: str, out: Tee) -> None:
    name = f"drift-demo-{label}-{time.strftime('%H%M%S')}"
    kubectl("create", "job", name, "--from=cronjob/drift-monitor")
    kube.wait_job(name, 600)
    out(f"$ kubectl logs job/{name}")
    out(kubectl("logs", f"job/{name}", capture=True).stdout.rstrip())


def alerts(url: str) -> list[dict]:
    status, body = kube.http("GET", f"{url}/api/v2/alerts?active=true&silenced=false")
    return body if status == 200 and isinstance(body, list) else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--wait", type=int, default=240, help="seconds to wait for the alert")
    ap.add_argument("--shift-std", default="3.0")
    ap.add_argument("--fraction", default="0.4")
    ap.add_argument("--columns", default="sigma")
    args = ap.parse_args()
    out = Tee()

    with kube.forward("svc/ml-service", 8000) as url:
        out("== 1. baseline window: unchanged rows ==")
        replay(url, "baseline", out, [])
        monitor_run("baseline", out)
        out("== 2. current window with the injected shift ==")
        replay(url, "shift", out, ["--shift-columns", args.columns, "--shift-std", args.shift_std,
                                   "--fraction", args.fraction])
        monitor_run("shift", out)

    out("== 3. the alert ==")
    seen: list[dict] = []
    with kube.forward("svc/alertmanager", 9093) as am:
        deadline = time.monotonic() + args.wait
        while time.monotonic() < deadline:
            seen = alerts(am)
            if any(a["labels"]["alertname"] == "FeatureDriftSigma" for a in seen):
                break
            time.sleep(10)
    for a in seen:
        out(f"alert {a['labels']['alertname']} ({a['labels'].get('severity', '?')}), "
            f"starts {a['startsAt']}: {a['annotations'].get('summary', '')[:160]}")
    if not any(a["labels"]["alertname"] == "FeatureDriftSigma" for a in seen):
        out("FeatureDriftSigma did not fire within the wait: check Prometheus (:9090, Alerts)")
        return 1

    out("== 4. the responder started a retraining Job (stretch) ==")
    time.sleep(10)
    jobs = kubectl("get", "jobs", "-l", "app=retrain", "-o", "wide", capture=True).stdout.rstrip()
    out("$ kubectl get jobs -l app=retrain\n" + jobs)
    out("$ kubectl logs deploy/responder --tail=8")
    out(kubectl("logs", "deploy/responder", "--tail=8", capture=True).stdout.rstrip())
    names = json.loads(kubectl("get", "jobs", "-l", "app=retrain", "-o", "json", capture=True).stdout)
    if names["items"]:
        job = names["items"][-1]["metadata"]["name"]
        out(f"annotation: {names['items'][-1]['metadata']['annotations']}")
        kube.wait_job(job, 600)
        out(f"$ kubectl logs job/{job}")
        out(kubectl("logs", f"job/{job}", capture=True).stdout.rstrip())
    out.save(ROOT / "docs" / "evidence" / "drift-demo.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
