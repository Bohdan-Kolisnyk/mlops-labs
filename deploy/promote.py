"""Promote a model version by moving the alias, with no image rebuild (the proof of lab 4, section 3).

    python deploy/promote.py --new-version          # train another version in the cluster, promote it
    python deploy/promote.py --version 3            # promote an existing version
    python deploy/promote.py --rollback             # champion goes back to the version `previous` names

Steps: set the alias `champion` in the registry (the old champion keeps the alias `previous`),
restart the ML Deployment, ask /predict which model version answers. The image tag does not change:
the model comes from the registry by alias, the image holds only code.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kube
from kube import ROOT, Tee, kubectl

SET_ALIAS = (
    "import sys; from mlflow import MlflowClient; c = MlflowClient('http://localhost:5000'); "
    "n, a, v = sys.argv[1:4]; c.set_registered_model_alias(n, a, v); "
    "print('alias', a, '->', c.get_model_version_by_alias(n, a).version)"
)
READ_ALIAS = (
    "import sys; from mlflow import MlflowClient; c = MlflowClient('http://localhost:5000'); "
    "n, a = sys.argv[1:3]; print(c.get_model_version_by_alias(n, a).version)"
)
MODEL = "truth-social-abnormal-move"


def alias_version(alias: str) -> str | None:
    out = kubectl("exec", "deploy/mlflow", "--", "python", "-c", READ_ALIAS, MODEL, alias,
                  capture=True, check=False)
    return out.stdout.strip().splitlines()[-1] if out.returncode == 0 and out.stdout.strip() else None


def ask_service(out: Tee, label: str) -> None:
    with kube.forward("svc/ml-service", 8000) as url:
        status, health = kube.http("GET", f"{url}/health")
        status2, body = kube.http("POST", f"{url}/predict", {"instances": [kube.sample_row()]})
        out(f"{label}: GET /health -> {status} {health}")
        out(f"{label}: POST /predict -> {status2}, model_version={body['model_version']}, "
            f"alias={body['alias']}, image_tag={body['image_tag']}, "
            f"probability={body['predictions'][0]['probability']:.4f}")
    out(f"{label}: deployment image = {kube.deployment_image('ml-service')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--new-version", action="store_true")
    group.add_argument("--version")
    group.add_argument("--rollback", action="store_true")
    ap.add_argument("--c", default="0.3", help="C of the new version (--new-version)")
    ap.add_argument("--save", default=str(ROOT / "docs" / "evidence" / "alias-promotion.txt"))
    args = ap.parse_args()
    out = Tee()
    out(f"$ python deploy/promote.py {' '.join(sys.argv[1:])}")
    image = kube.deployment_image("ml-service")

    out("== before ==")
    old = alias_version("champion")
    out(f"registry: champion -> version {old}")
    ask_service(out, "before")

    if args.new_version:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        log = kube.run_job(f"model-train-{stamp}", image,
                           ["mlops_labs.serve.bootstrap", "--c", args.c, "--alias", "none"],
                           "model-train")
        line = next((ln for ln in log.splitlines() if ln.startswith("registered")), log[-300:])
        out(f"== new version trained in the cluster ==\n{line}")
        target = line.split("version")[1].split(",")[0].strip()
    elif args.rollback:
        target = alias_version("previous")
        if target is None:
            sys.exit("the alias `previous` does not exist: nothing to roll back to")
    else:
        target = args.version

    out(f"== promote version {target} ==")
    if target == old:
        sys.exit(f"champion is already version {old}")
    result = kubectl("exec", "deploy/mlflow", "--", "python", "-c", SET_ALIAS, MODEL, "previous", old,
                     capture=True)
    out(result.stdout.strip().splitlines()[-1])
    result = kubectl("exec", "deploy/mlflow", "--", "python", "-c", SET_ALIAS, MODEL, "champion", target,
                     capture=True)
    out(result.stdout.strip().splitlines()[-1])

    out("== restart the Deployment (no image rebuild) ==")
    kubectl("rollout", "restart", "deployment/ml-service")
    out(kubectl("rollout", "status", "deployment/ml-service", "--timeout=300s", capture=True).stdout.strip())
    ask_service(out, "after")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    kube.run_job(f"reference-refresh-{stamp}", image,
                 ["mlops_labs.serve.bootstrap", "--refresh-reference"], "reference-refresh")
    out("reference window rewritten for the new champion (the drift job compares like with like)")
    out(f"image tag unchanged: {image}")
    out.save(Path(args.save), append=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
