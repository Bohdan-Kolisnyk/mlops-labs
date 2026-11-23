"""Answer quality beyond the reference score: groundedness, hallucination rate, answer relevance,
context completeness and context utilization.

Three judge prompts (files, hashed in every run). Groundedness splits the answer into claims and
checks each against the numbered passages the answer was given; hallucination rate and context
utilization come from the same call. Relevance reads only question and answer. Completeness reads
the passages and the reference answer, so it measures the retriever, not the generator.
"""

from __future__ import annotations

import numpy as np

from rag_lab import prompts
from rag_lab.core import claim_stats, quality_summary
from rag_lab.llm import ask_many, usage
from rag_lab.retrieval import numbered_context

PROMPT_NAMES = ("groundedness", "answer_relevance", "context_completeness")


def _score(reply, low: int, high: int) -> tuple[float | None, str | None]:
    obj = prompts.parse_json(reply.text) or {}
    s = obj.get("score")
    if isinstance(s, (int, float)) and not isinstance(s, bool) and low <= s <= high:
        return float(s), obj.get("reason")
    return None, obj.get("reason")


def _claims(reply) -> tuple[bool, list[dict]] | None:
    obj = prompts.parse_json(reply.text) or {}
    claims = obj.get("claims")
    if not isinstance(claims, list) or not all(isinstance(c, dict) for c in claims):
        return None
    return obj.get("declines") is True, claims


class _Limited:
    def __init__(self, llm, max_tokens: int):
        self.llm, self.max_tokens = llm, max_tokens

    def ask(self, prompt: str):
        return self.llm.ask(prompt, self.max_tokens)


def judge_quality(cfg: dict, llm, questions: list[dict], answers: list[str], hits: list[list[dict]]) -> dict:
    """Per-question fields and run-level metrics for one top-k value."""
    workers = cfg["llm"]["concurrency"]
    judge = _Limited(llm, cfg["llm"]["judge_max_tokens"])
    tpl = {name: prompts.load(cfg, name)[0] for name in PROMPT_NAMES}
    passages = [numbered_context(h) for h in hits]

    g_in = [
        prompts.render(tpl["groundedness"], passages=p, question=q["question"], answer=a)
        for p, q, a in zip(passages, questions, answers, strict=True)
    ]
    r_in = [
        prompts.render(tpl["answer_relevance"], question=q["question"], answer=a)
        for q, a in zip(questions, answers, strict=True)
    ]
    c_in = [
        prompts.render(tpl["context_completeness"], passages=p, question=q["question"], reference=q["reference_answer"])
        for p, q in zip(passages, questions, strict=True)
    ]
    g_rep, r_rep, c_rep = (ask_many(judge, batch, workers) for batch in (g_in, r_in, c_in))

    stats, rows, relevance, answered_relevance, completeness = [], [], [], [], []
    for g, r, c, h in zip(g_rep, r_rep, c_rep, hits, strict=True):
        parsed = _claims(g)
        declines, claims = (None, None) if parsed is None else parsed
        s = None if parsed is None else claim_stats([] if declines else claims, len(h))
        rel, rel_reason = _score(r, 1, 5)
        com, com_reason = _score(c, 0, 2)
        if s is not None:
            stats.append(s)
        if rel is not None:
            relevance.append(rel)
            if s is not None and s["claims"] > 0:
                answered_relevance.append(rel)
        if com is not None:
            completeness.append(com)
        rows.append(
            {
                "declines": declines,
                "claims": claims,
                "groundedness": None if s is None else s["groundedness"],
                "hallucinated": None if s is None else s["hallucinated"],
                "context_utilization": None if s is None else s["utilization"],
                "answer_relevance": rel,
                "answer_relevance_reason": rel_reason,
                "context_completeness": com,
                "context_completeness_reason": com_reason,
                "judge_stop_reasons": [g.stop_reason, r.stop_reason, c.stop_reason],
                "judge_tokens_out": [g.tokens_out, r.tokens_out, c.tokens_out],
                "groundedness_raw": g.text[:300] if claims is None else None,
            }
        )

    use = usage(g_rep + r_rep + c_rep)
    metrics = quality_summary(stats)
    metrics.update(
        answer_relevance=float(np.mean(relevance)) / 5 if relevance else float("nan"),
        answer_relevance_answered=float(np.mean(answered_relevance)) / 5 if answered_relevance else float("nan"),
        context_completeness=float(np.mean(completeness)) / 2 if completeness else float("nan"),
        groundedness_parse_failures=len(rows) - len(stats),
        relevance_parse_failures=len(rows) - len(relevance),
        completeness_parse_failures=len(rows) - len(completeness),
        quality_parse_failures=3 * len(rows) - len(stats) - len(relevance) - len(completeness),
        judges_not_finished=sum(x.stop_reason != "end_turn" for x in g_rep + r_rep + c_rep),
        quality_calls=use["calls"],
        quality_tokens_in=use["tokens_in"],
        quality_tokens_out=use["tokens_out"],
        quality_cost_usd=use["cost_usd"],
    )
    return {"rows": rows, "metrics": metrics}
