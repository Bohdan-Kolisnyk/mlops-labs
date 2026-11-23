"""Keyword retrieval: Okapi BM25 over the chunks in the index, and reciprocal rank fusion with the
dense ranking. The chunk text starts with the post date; the date also becomes a month name token,
so a question about "December 2025" has something to match that the embedding model lacks."""

from __future__ import annotations

import math
import re

import numpy as np

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "had",
        "has", "have", "he", "her", "his", "how", "in", "is", "it", "its", "of", "on", "or", "she",
        "that", "the", "their", "they", "this", "to", "was", "were", "what", "when", "where",
        "which", "who", "whom", "why", "with", "would",
    }
)
DATE_PREFIX = re.compile(r"^\[(\d{4})-(\d{2})-\d{2}\]")


def tokens(text: str) -> list[str]:
    out = [w for w in re.findall(r"[\w']+", text.lower()) if w not in STOPWORDS]
    m = DATE_PREFIX.match(text)
    if m and 1 <= int(m.group(2)) <= 12:
        out.append(MONTHS[int(m.group(2)) - 1])
    return out


class BM25:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        docs = [tokens(c["text"]) for c in chunks]
        self.lengths = np.array([len(d) for d in docs], dtype=float)
        self.avg_len = float(self.lengths.mean()) if len(docs) else 0.0
        postings: dict[str, dict[int, int]] = {}
        for i, doc in enumerate(docs):
            for term in doc:
                postings.setdefault(term, {})
                postings[term][i] = postings[term].get(i, 0) + 1
        n = len(docs)
        self.index = {
            term: (
                np.fromiter(p.keys(), dtype=int, count=len(p)),
                np.fromiter(p.values(), dtype=float, count=len(p)),
                math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)),
            )
            for term, p in postings.items()
        }

    @classmethod
    def from_store(cls, store, params: dict) -> BM25:
        return cls(store.all_chunks(), params["k1"], params["b"])

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.chunks))
        for term in set(tokens(query)):
            if term not in self.index:
                continue
            docs, tf, idf = self.index[term]
            norm = self.k1 * (1 - self.b + self.b * self.lengths[docs] / self.avg_len)
            s[docs] += idf * tf * (self.k1 + 1) / (tf + norm)
        return s

    def search(self, query: str, k: int) -> list[dict]:
        s = self.scores(query)
        top = np.argsort(-s, kind="stable")[:k]
        return [{**self.chunks[i], "score": float(s[i])} for i in top if s[i] > 0]


def rrf(rankings: list[list[dict]], k: int, rrf_k: int = 60) -> list[dict]:
    """Reciprocal rank fusion: each ranking gives 1 / (rrf_k + rank) to a chunk, the sums decide."""
    total: dict[str, float] = {}
    first: dict[str, dict] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            cid = hit["chunk_id"]
            total[cid] = total.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            first.setdefault(cid, hit)
    order = sorted(total, key=lambda c: (-total[c], c))[:k]
    return [{**first[c], "score": total[c]} for c in order]
