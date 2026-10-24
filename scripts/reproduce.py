"""Reproduce a registered model from its identifiers with one command.

    uv run python scripts/reproduce.py models:/truth-social-abnormal-move@champion
    uv run python scripts/reproduce.py <run_id>            # a run instead of a model version

What it does:
  1. resolves the model version (or run) in the MLflow registry and reads its two lineage
     tags, `data_version` and `git_commit`;
  2. checks that the code in this checkout is the code of that commit (src/, scripts/,
     pyproject.toml, uv.lock), and refuses with the git command to run if it is not;
  3. makes data/raw match the data version: `dvc pull`, and if data/raw.dvc names another
     version it is restored from the commit first (and put back afterwards);
  4. trains again from data/raw with the parameters of the original run;
  5. compares val and test PR-AUC with the original run and exits 0 when both differ by
     at most the tolerance (default 0.005), 1 otherwise.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlflow import MlflowClient

from mlops_labs import lineage
from mlops_labs.data import read_data_version
from mlops_labs.env import ROOT, load_env
from mlops_labs.train import CHECKED_METRICS, MODEL_NAME, run_training

DEFAULT_TOLERANCE = 0.005
DVC = [sys.executable, "-m", "dvc"]


def sh(*cmd: str, check: bool = True) -> str:
    out = subprocess.run(
        cmd, check=False, cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
    )
    if check and out.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{out.stdout}\n{out.stderr}")
    return out.stdout.strip()


def sh_rc(*cmd: str) -> int:
    return subprocess.run(cmd, check=False, cwd=ROOT, capture_output=True).returncode


def resolve(client: MlflowClient, ref: str) -> tuple[str, dict[str, str], str]:
    """(run_id, lineage tags, description of what was resolved)."""
    if ref.startswith("models:/"):
        spec = ref.removeprefix("models:/")
        name, _, selector = spec.partition("@")
        if not selector:
            name, _, selector = spec.partition("/")
        mv = (
            client.get_model_version_by_alias(name, selector)
            if "@" in spec
            else client.get_model_version(name, selector)
        )
        tags = dict(mv.tags)
        return mv.run_id, tags, f"{name} version {mv.version}"
    run = client.get_run(ref)
    return run.info.run_id, dict(run.data.tags), f"run {ref}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("ref", nargs="?", default=f"models:/{MODEL_NAME}@champion")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    ap.add_argument("--no-pull", action="store_true", help="skip dvc pull (data already there)")
    args = ap.parse_args()

    load_env()
    client = MlflowClient()
    run_id, tags, what = resolve(client, args.ref)
    data_version, commit = tags.get("data_version"), tags.get("git_commit")
    if not data_version or not commit:
        raise SystemExit(f"{what} has no data_version/git_commit tags: it cannot be reproduced")
    run = client.get_run(run_id)
    print(f"{what}\n  run        {run_id}\n  data       {data_version}\n  commit     {commit}")

    # code: same files as at the commit
    if lineage.git_dirty():
        raise SystemExit("uncommitted changes in src/, scripts/, pyproject.toml or uv.lock")
    differs = sh("git", "diff", "--name-only", commit, "HEAD", "--", *lineage.CODE_PATHS)
    if differs:
        raise SystemExit(
            f"the code differs from commit {commit[:10]} in:\n  {differs}\n"
            f"run `git checkout {commit}`, then this command again"
        )

    # data: data/raw must be the version named by the tag
    restore = False
    if read_data_version(ROOT / "data" / "raw") != data_version:
        old = sh("git", "show", f"{commit}:data/raw.dvc")
        if f"md5: {data_version}" not in old:
            raise SystemExit(f"data/raw.dvc at commit {commit[:10]} does not name {data_version}")
        sh("git", "checkout", commit, "--", "data/raw.dvc")
        restore = True
    try:
        if not args.no_pull:
            sh(*DVC, "pull", "data/raw.dvc")
        if read_data_version(ROOT / "data" / "raw") != data_version:
            raise SystemExit("data/raw.dvc does not match the data_version tag after checkout")
        if sh_rc(*DVC, "status", "--quiet", "data/raw.dvc") != 0:
            raise SystemExit(
                "data/raw differs from the version in data/raw.dvc: run `dvc checkout data/raw.dvc`"
            )
        params = run.data.params
        if "C" not in params:
            raise SystemExit(f"run {run_id} is not the candidate run (no parameter C)")
        result = run_training(
            source="raw",
            c=float(params["C"]),
            seed=int(params["seed"]),
            bootstrap_n=int(params["bootstrap_n"]),
            extra_tags={"reproduces_run": run_id},
        )
    finally:
        if restore:
            sh("git", "checkout", "HEAD", "--", "data/raw.dvc")
            if not args.no_pull:
                sh(*DVC, "checkout", "data/raw.dvc", check=False)

    new = result["metrics"]["candidate-tfidf-lr"]
    ok = True
    print(f"\ntolerance: {args.tolerance} (absolute, PR-AUC)")
    for name in CHECKED_METRICS:
        old_v = run.data.metrics[name]
        diff = abs(new[name] - old_v)
        flag = "ok" if diff <= args.tolerance else "FAIL"
        ok &= diff <= args.tolerance
        print(
            f"  {name:14} original {old_v:.5f}  reproduced {new[name]:.5f}  diff {diff:.5f}  {flag}"
        )
    print("REPRODUCED" if ok else "NOT REPRODUCED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
