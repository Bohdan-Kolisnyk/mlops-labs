"""Baselines B0-B2 and the candidate of design 5.3, and the offline metrics of design 5.4."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_labs.features import FEATURE_COLUMNS

TOP_RATE = 0.20  # alert rate of the operating point (design 3)
N_SLOTS = 5

# B1: fixed a priori from the topic of the project, not tuned on the labels (design 5.3).
KEYWORDS = [
    "tariff",
    "tariffs",
    "china",
    "chinese",
    "fed",
    "federal reserve",
    "powell",
    "interest rate",
    "interest rates",
    "inflation",
    "recession",
    "economy",
    "stock market",
    "market",
    "markets",
    "dollar",
    "oil",
    "tax",
    "taxes",
    "trade",
    "deal",
    "sanctions",
    "debt",
    "treasury",
    "bank",
    "banks",
    "iran",
    "russia",
    "ukraine",
    "war",
]
_KEYWORD_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in KEYWORDS) + r")\b", re.IGNORECASE)


def keywords_hash() -> str:
    return hashlib.sha256("\n".join(KEYWORDS).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------- models


class ConstantPrior:
    """B0: every post gets the base rate of the training set."""

    name = "B0-prior"

    def fit(self, df: pd.DataFrame) -> ConstantPrior:
        self.prior_ = float(df["y"].mean())
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.prior_)


class KeywordRule:
    """B1: the number of keyword hits in the text. Nothing is learned."""

    name = "B1-keywords"

    def fit(self, df: pd.DataFrame) -> KeywordRule:
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return df["text"].map(lambda t: len(_KEYWORD_RE.findall(t))).to_numpy(dtype=float)


class ContextOnly:
    """B2: logistic regression on the time slot, sigma and the last completed bar. No text."""

    name = "B2-context"

    def __init__(self, c: float = 1.0):
        self.c = c

    @staticmethod
    def _matrix(df: pd.DataFrame) -> np.ndarray:
        slots = np.eye(N_SLOTS)[df["slot_index"].astype(int).to_numpy()]
        last = df["last_bar_return"].fillna(0.0).to_numpy()
        return np.column_stack([slots, df["sigma"].to_numpy(), last, np.abs(last)])

    def fit(self, df: pd.DataFrame) -> ContextOnly:
        self.scaler_ = StandardScaler().fit(self._matrix(df))
        self.model_ = LogisticRegression(C=self.c, max_iter=1000).fit(
            self.scaler_.transform(self._matrix(df)), df["y"]
        )
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self.scaler_.transform(self._matrix(df)))[:, 1]


CANDIDATE_INPUT = ["text", *FEATURE_COLUMNS]


def build_candidate(c: float = 1.0, seed: int = 0) -> Pipeline:
    """TF-IDF of the text plus the scaled context features into a logistic regression.

    The vocabulary is cut by `min_df` only, never by `max_features`: with a top-k cut, many
    terms tie at the boundary (here 231 terms with the same count) and which of them stay
    depends on the sort implementation of the CPU, so two machines got different models.
    """
    prep = ColumnTransformer(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True),
                "text",
            ),
            ("num", StandardScaler(), FEATURE_COLUMNS),
        ]
    )
    clf = LogisticRegression(C=c, max_iter=2000, random_state=seed)
    return Pipeline([("prep", prep), ("clf", clf)])


def prepare_candidate_input(df: pd.DataFrame) -> pd.DataFrame:
    """The columns the candidate expects; the only NaN that can occur is the first bar."""
    x = df[CANDIDATE_INPUT].copy()
    x["last_bar_return"] = x["last_bar_return"].fillna(0.0)
    # all numeric inputs as float64: the model signature then accepts any value at serving
    x[FEATURE_COLUMNS] = x[FEATURE_COLUMNS].astype("float64")
    return x


class Candidate:
    name = "candidate-tfidf-lr"

    def __init__(self, c: float = 1.0, seed: int = 0):
        self.c, self.seed = c, seed

    def fit(self, df: pd.DataFrame) -> Candidate:
        self.pipeline_ = build_candidate(self.c, self.seed).fit(
            prepare_candidate_input(df), df["y"]
        )
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(prepare_candidate_input(df))[:, 1]


# --------------------------------------------------------------------------- metrics


def jitter(scores: np.ndarray, seed: int) -> np.ndarray:
    """Break ties at random (seeded), so 'the top 20%' is exactly 20% for every model.

    Needed for B0 (all scores equal) and B1 (most posts have zero hits): the alerts that the
    rule cannot rank are filled at random, which is what a rule-based fallback would do.
    """
    rng = np.random.default_rng(seed)
    return scores + 1e-9 * rng.random(len(scores))


def alert_threshold(val_scores: np.ndarray, seed: int, rate: float = TOP_RATE) -> float:
    """Score above which `rate` of the validation posts fall (jittered). Fixed afterwards."""
    return float(np.quantile(jitter(val_scores, seed), 1 - rate))


def operating_point(
    y: np.ndarray, scores: np.ndarray, threshold: float, seed: int
) -> dict[str, float]:
    alert = jitter(scores, seed) >= threshold
    positives = max(int(y.sum()), 1)
    caught = int((alert & (y == 1)).sum())
    return {
        "alert_rate": float(alert.mean()),
        "coverage": caught / positives,
        "precision": caught / max(int(alert.sum()), 1),
    }


def clustered_bootstrap_ci(
    y: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    clusters: np.ndarray,
    n: int,
    seed: int,
    reference: str | None = None,
) -> dict[str, tuple[float, float]]:
    """95% bootstrap interval of PR-AUC per model, resampling whole hours (design 5.4).

    With `reference`, also the interval of PR-AUC(model) - PR-AUC(reference), paired on the
    same resamples, under the key '<model>-<reference>'.
    """
    codes, uniques = pd.factorize(clusters)
    members = [np.flatnonzero(codes == i) for i in range(len(uniques))]
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in scores_by_model}
    diffs: dict[str, list[float]] = {}
    if reference:
        diffs = {m: [] for m in scores_by_model if m != reference}
    done = 0
    while done < n:
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if y[idx].sum() == 0:
            continue
        ap = {m: average_precision_score(y[idx], s[idx]) for m, s in scores_by_model.items()}
        for m, v in ap.items():
            draws[m].append(v)
        for m in diffs:
            diffs[m].append(ap[m] - ap[reference])
        done += 1
    out = {
        m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in draws.items()
    }
    for m, v in diffs.items():
        out[f"{m}-{reference}"] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


def point_metrics(y: np.ndarray, scores: np.ndarray, probabilistic: bool) -> dict[str, float]:
    m = {
        "pr_auc": float(average_precision_score(y, scores)),
        "base_rate": float(y.mean()),
    }
    if len(np.unique(scores)) > 1:
        m["roc_auc"] = float(roc_auc_score(y, scores))
    if probabilistic:
        m["brier"] = float(brier_score_loss(y, np.clip(scores, 0, 1)))
    return m
