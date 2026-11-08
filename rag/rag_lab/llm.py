"""The LLM, called through the Anthropic API. Model name, token limit and prices
are read from config.yaml; the key is read from ANTHROPIC_API_KEY (never from the repository)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class Reply:
    text: str
    tokens_in: int
    tokens_out: int
    seconds: float
    cost: float


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
        return Reply(text, tin, tout, seconds, cost)


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
    }
