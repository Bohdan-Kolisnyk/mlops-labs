"""The LLM, called through the Anthropic API or through a self-hosted OpenAI-compatible server
(Ollama in this repository, vLLM would do as well). Model names, addresses, token limits and prices are read from config.yaml; the Anthropic
key is read from ANTHROPIC_API_KEY (never from the repository)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class Reply:
    text: str
    tokens_in: int
    tokens_out: int
    seconds: float
    cost: float
    stop_reason: str = "end_turn"


class Claude:
    def __init__(self, cfg: dict):
        import anthropic

        self.cfg = cfg["llm"]
        self.client = anthropic.Anthropic()
        self._anthropic = anthropic

    def ask(self, prompt: str, max_tokens: int | None = None) -> Reply:
        c = self.cfg
        kwargs = {
            "model": c["model"],
            "max_tokens": max_tokens or c["max_tokens"],
            "messages": [{"role": "user", "content": prompt}],
        }
        t0 = time.perf_counter()
        # no temperature: the current models do not accept sampling parameters, so every run
        # uses the model's default; the model id is logged in every run for that reason
        msg = self.client.messages.create(**kwargs)
        seconds = time.perf_counter() - t0
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        tin, tout = msg.usage.input_tokens, msg.usage.output_tokens
        cost = (tin * c["price_per_mtok_in"] + tout * c["price_per_mtok_out"]) / 1e6
        return Reply(text, tin, tout, seconds, cost, str(msg.stop_reason))


class LocalLLM:
    """A self-hosted model behind an OpenAI-compatible chat endpoint (Ollama and vLLM serve one).

    There is no price per token. The cost of a run is the wall-clock time of its calls times a
    shadow price per GPU hour (an assumption in config.yaml), computed by the evaluation."""

    def __init__(self, cfg: dict):
        self.cfg = cfg["llm_local"]
        self.base_url = self.cfg["base_url"].rstrip("/")
        self.gpu_usd_per_hour = float(self.cfg["gpu_usd_per_hour"])

    def ask(self, prompt: str, max_tokens: int | None = None) -> Reply:
        c = self.cfg
        body = {
            "model": c["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or c["max_tokens"],
            "temperature": c["temperature"],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=c["timeout_s"]) as response:
                data = json.load(response)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"cannot reach the local LLM server at {self.base_url} ({exc.reason}): "
                "start the model server (README, Lab 3, self-hosted model) and wait for the model to load"
            ) from exc
        seconds = time.perf_counter() - t0
        choice = data["choices"][0]
        usage_ = data.get("usage", {})
        stop = "end_turn" if choice.get("finish_reason") == "stop" else "max_tokens"
        return Reply(
            choice["message"].get("content") or "",
            int(usage_.get("prompt_tokens", 0)),
            int(usage_.get("completion_tokens", 0)),
            seconds,
            0.0,
            stop,
        )

    def server_version(self) -> str:
        root = self.base_url.rsplit("/v1", 1)[0]
        for path in ("/api/version", "/version"):
            try:
                with urllib.request.urlopen(root + path, timeout=10) as response:
                    return str(json.load(response).get("version", "unknown"))
            except (urllib.error.URLError, ValueError, OSError):
                continue
        return "unknown"


def make_llm(cfg: dict, which: str = "anthropic"):
    if which == "anthropic":
        return Claude(cfg)
    if which == "local":
        return LocalLLM(cfg)
    raise ValueError(f"unknown LLM {which!r}")


def model_name(cfg: dict, which: str = "anthropic") -> str:
    return cfg["llm"]["model"] if which == "anthropic" else cfg["llm_local"]["model"]


def ask_many(llm, prompts: list[str], workers: int = 6) -> list[Reply]:
    """Replies in the order of the prompts; each call keeps its own latency."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(llm.ask, prompts))


def usage(replies: list[Reply]) -> dict[str, float]:
    from rag_lab.core import percentiles

    p = percentiles([r.seconds for r in replies])
    return {
        "calls": len(replies),
        "tokens_in": sum(r.tokens_in for r in replies),
        "tokens_out": sum(r.tokens_out for r in replies),
        "cost_usd": sum(r.cost for r in replies),
        "latency_p50_s": p["p50"],
        "latency_p95_s": p["p95"],
        "latency_max_s": max(r.seconds for r in replies),
    }
