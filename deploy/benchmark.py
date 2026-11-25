"""Measured performance of the deployed services (design 6.3 and 7.3).

    python deploy/benchmark.py

ML service: single-row latency (500 sequential requests), batch throughput (20 batches of 50 rows)
and latency under 4 concurrent clients. LLM service: 10 questions of the Lab 3 evaluation set, one
after another (latency of retrieval plus the API call, tokens, cost). Memory and CPU of every pod
from `kubectl top` when the metrics-server addon is enabled (`minikube addons enable metrics-server`).
Prints markdown tables and saves them to docs/evidence/benchmark.md.
"""

from __future__ import annotations

import http.client
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kube
from kube import ROOT, Tee, kubectl


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


_local = threading.local()
RETRIES = {"count": 0}


def timed_post(url: str, body: dict) -> float:
    parsed = urlparse(url)
    payload = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    t0 = time.perf_counter()
    for attempt in range(2):
        conn = getattr(_local, "conn", None)
        if conn is None:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
            _local.conn = conn
        try:
            conn.request("POST", parsed.path, payload, headers)
            response = conn.getresponse()
            raw = response.read()
            status = response.status
            break
        except (OSError, http.client.HTTPException):
            conn.close()
            _local.conn = None
            if attempt == 1:
                raise
            RETRIES["count"] += 1
            t0 = time.perf_counter()
    if status != 200:
        raise RuntimeError(f"{url}: {status} {raw[:200]!r}")
    return time.perf_counter() - t0


def main() -> int:
    out = Tee()
    row = kube.sample_row()
    with kube.forward("svc/ml-service", 8000) as url:
        for _ in range(20):
            timed_post(f"{url}/predict", {"instances": [row]})
        single = [timed_post(f"{url}/predict", {"instances": [row]}) for _ in range(500)]
        t0 = time.perf_counter()
        batch = [timed_post(f"{url}/predict", {"instances": [row] * 50}) for _ in range(20)]
        batch_wall = time.perf_counter() - t0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            concurrent = list(pool.map(lambda _: timed_post(f"{url}/predict", {"instances": [row]}),
                                       range(300)))
        concurrent_wall = time.perf_counter() - t0
    out("### ML service (through kubectl port-forward, so every number includes the forward)")
    out("")
    out(f"Persistent HTTP connections; requests retried after a dropped forward: {RETRIES['count']} "
        "(the retry is timed from its own start).")
    out("")
    out("| scenario | requests | p50 ms | p95 ms | p99 ms | throughput |")
    out("|---|---|---|---|---|---|")
    out(f"| 1 row, sequential | {len(single)} | {1000 * pct(single, .5):.1f} | "
        f"{1000 * pct(single, .95):.1f} | {1000 * pct(single, .99):.1f} | "
        f"{len(single) / sum(single):.0f} req/s |")
    out(f"| 50 rows per request | {len(batch)} | {1000 * pct(batch, .5):.1f} | "
        f"{1000 * pct(batch, .95):.1f} | {1000 * pct(batch, .99):.1f} | "
        f"{50 * len(batch) / batch_wall:.0f} rows/s |")
    out(f"| 1 row, 4 clients | {len(concurrent)} | {1000 * pct(concurrent, .5):.1f} | "
        f"{1000 * pct(concurrent, .95):.1f} | {1000 * pct(concurrent, .99):.1f} | "
        f"{len(concurrent) / concurrent_wall:.0f} req/s |")

    questions = [json.loads(ln)["question"] for ln in
                 (ROOT / "rag" / "eval" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines()[:10]]
    with kube.forward("svc/llm-service", 8000) as url:
        status, health = kube.http("GET", f"{url}/health")
        out("")
        if status != 200:
            out(f"### LLM service: not ready ({health}), skipped")
        else:
            rows = []
            for q in questions:
                t0 = time.perf_counter()
                status, body = kube.http("POST", f"{url}/ask", {"question": q}, timeout=180)
                if status == 200:
                    rows.append((time.perf_counter() - t0, body))
            totals = [r[0] for r in rows]
            out(f"### LLM service ({health['model']}, top-k from rag/config.yaml, {len(rows)} questions)")
            out("")
            out("| metric | value |")
            out("|---|---|")
            out(f"| latency p50 / p95 / max (s) | {pct(totals, .5):.2f} / {pct(totals, .95):.2f} / {max(totals):.2f} |")
            out(f"| of which retrieval, mean (s) | {statistics.mean(r[1]['retrieval_s'] for r in rows):.3f} |")
            out(f"| of which answer call, mean (s) | {statistics.mean(r[1]['generation_s'] for r in rows):.2f} |")
            out(f"| tokens in / out per request, mean | {statistics.mean(r[1]['tokens_in'] for r in rows):.0f} / "
                f"{statistics.mean(r[1]['tokens_out'] for r in rows):.0f} |")
            out(f"| cost per request (USD), mean | {statistics.mean(r[1]['cost_usd'] for r in rows):.5f} |")

    out("")
    top = kubectl("top", "pods", capture=True, check=False)
    if top.returncode == 0:
        out("### Pods (kubectl top)")
        out("")
        out("```text")
        out(top.stdout.rstrip())
        out("```")
    else:
        out("kubectl top is not available: minikube addons enable metrics-server, wait a minute, run again")
    out.save(ROOT / "docs" / "evidence" / "benchmark.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
