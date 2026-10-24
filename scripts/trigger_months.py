"""Trigger the ETL DAG once per month, in order (runs queue: max_active_runs=1).

    uv run python scripts/trigger_months.py                        # 2023-12 ... 2026-10
    uv run python scripts/trigger_months.py --from 2025-03 --to 2025-05

Arguments go to docker as a list, so the JSON quotes survive on Windows and Linux alike.
"""

from __future__ import annotations

import argparse
import json
import subprocess

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--from", dest="first", default="2023-12")
    ap.add_argument("--to", dest="last", default="2026-10")
    ap.add_argument("--raw-dir", default=None, help="path inside the container, e.g. a broken copy")
    args = ap.parse_args()
    for month in pd.period_range(args.first, args.last, freq="M").strftime("%Y-%m"):
        conf = {"month": month}
        if args.raw_dir:
            conf["raw_dir"] = args.raw_dir
        subprocess.run(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "airflow-scheduler",
                "airflow",
                "dags",
                "trigger",
                "truth_social_etl",
                "--conf",
                json.dumps(conf),
            ],
            check=True,
            capture_output=True,
        )
        print(f"queued {month}")


if __name__ == "__main__":
    main()
