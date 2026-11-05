"""From a post to a score: the serving path of the champion model (Lab 2, end to end).

    text + time of the post
      -> bar and time slot of the post          (same rule as the labels)
      -> sigma of the slot from the feature store (Feast, point in time or online)
      -> the other features, built by the SAME function as in training (compute_features)
      -> champion model from the registry (loaded by alias)
      -> probability of an abnormal move and the alert decision (top 20% of the validation scores)

Feast lives in its own environment (feature_repo/, it pins pandas below 3), so sigma is
fetched by running `feature_repo/get_sigma.py` there and reading one JSON line from it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_labs import config, dataset, models
from mlops_labs.data import load_bars, load_posts, read_data_version
from mlops_labs.env import ROOT, load_env
from mlops_labs.features import compute_features

MODEL_NAME = "truth-social-abnormal-move"
MODEL_URI = f"models:/{MODEL_NAME}@champion"
FEATURE_REPO = ROOT / "feature_repo"
OUT_OF_WINDOW = "outside the 09:30-14:30 ET window of a trading day"


# ---------------------------------------------------------------------- feature store


def fetch_sigma(requests: list[dict], online: bool = False) -> list[float | None]:
    """sigma of a slot for each request {"slot": "10:30", "at": "<UTC time>"}.

    Default: offline store, point in time (the row valid at `at`, as in training).
    online=True: the newest value in the online store (what a live service would read).
    """
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("`uv` is not on PATH; it is needed to run the Feast environment")
    cmd = [uv, "run", "--project", str(FEATURE_REPO), "python", str(FEATURE_REPO / "get_sigma.py")]
    if online:
        cmd.append("--online")
    done = subprocess.run(
        cmd,
        input=json.dumps(requests),
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        check=False,
    )
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("[")]
    if done.returncode != 0 or not lines:
        raise RuntimeError(f"Feast lookup failed:\n{done.stdout}\n{done.stderr}")
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------- features


def locate_bars(created_at: pd.Series, bars: pd.DataFrame) -> pd.DataFrame:
    """The hourly bar that contains each post and its slot (the label rule, without the label)."""
    b = bars[["bar_start", "session_date", "slot"]].sort_values("bar_start")
    asked = pd.DataFrame({"created_at": created_at.to_numpy()}).sort_values("created_at")
    m = pd.merge_asof(
        asked,
        b,
        left_on="created_at",
        right_on="bar_start",
        direction="backward",
        tolerance=pd.Timedelta("60min"),
    )
    m.index = asked.index  # merge_asof needs sorted input; put the rows back in the caller's order
    return m.sort_index().reset_index(drop=True)


def serving_features(
    posts: pd.DataFrame,
    history: pd.DataFrame,
    bars: pd.DataFrame,
    online: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features of new posts, built like the training table. Returns (features, skipped).

    posts:   post_id, created_at (UTC), text, has_media
    history: earlier posts (post_id, created_at), for the number of posts in the last 60 minutes
    A post is skipped, with a reason, when it lies outside the window, has no text (the model
    was trained on posts with text) or when the feature store has no sigma for it.
    """
    posts = posts.reset_index(drop=True).copy()
    where = locate_bars(posts["created_at"], bars)
    posts["bar_start"] = where["bar_start"]
    posts["slot"] = where["slot"]
    posts["session_date"] = where["session_date"]

    reason = pd.Series("", index=posts.index, dtype="object")
    reason[~posts["slot"].isin(config.POST_SLOTS)] = OUT_OF_WINDOW
    reason[(reason == "") & (posts["text"].str.strip() == "")] = "no text (media or repost)"

    ok = reason == ""
    posts["sigma"] = np.nan
    if ok.any():
        asked = [
            {"slot": s, "at": t.isoformat()}
            for s, t in zip(posts.loc[ok, "slot"], posts.loc[ok, "created_at"], strict=True)
        ]
        posts.loc[ok, "sigma"] = [np.nan if v is None else v for v in fetch_sigma(asked, online)]
        reason[ok & posts["sigma"].isna()] = "the feature store has no sigma for this slot and time"

    ok = reason == ""
    skipped = posts.loc[~ok, ["post_id", "created_at"]].assign(reason=reason[~ok])
    if not ok.any():
        return pd.DataFrame(), skipped.reset_index(drop=True)

    clean = posts[ok].assign(text_empty=False, dup_text=False)
    feats = compute_features(clean, history, bars)
    return feats, skipped.reset_index(drop=True)


def candidate_input(feats: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    """The exact columns the candidate model expects, in the order of the posts."""
    df = feats.merge(posts[["post_id", "text"]], on="post_id", validate="one_to_one")
    return models.prepare_candidate_input(df)


# ------------------------------------------------------------------------ model, alert


def load_model(model_uri: str = MODEL_URI):
    """The champion as a scikit-learn pipeline, resolved by alias in the registry."""
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    load_env()
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    if model_uri.startswith("models:/") and "@" in model_uri:
        name, alias = model_uri.removeprefix("models:/").split("@")
        mv = MlflowClient().get_model_version_by_alias(name, alias)
        info = {
            "version": mv.version,
            "data_version": mv.tags.get("data_version", "?"),
            "git_commit": mv.tags.get("git_commit", "?"),
        }
    else:
        info = {"version": "?", "data_version": "?", "git_commit": "?"}
    return mlflow.sklearn.load_model(model_uri), info


def alert_threshold(model, raw_dir: str | Path) -> float:
    """Score above which 20% of the validation posts fall (the operating point of design 3)."""
    table, _ = dataset.from_raw(raw_dir)
    val = table[dataset.split_name(table["session_date"]) == "val"]
    scores = model.predict_proba(models.prepare_candidate_input(val))[:, 1]
    return float(np.quantile(scores, 1 - models.TOP_RATE))


# ---------------------------------------------------------------------------- scoring


def score_posts(
    posts: pd.DataFrame,
    raw_dir: str | Path,
    model=None,
    threshold: float | None = None,
    online: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Probability and alert flag for every scoreable post; the rest come back with a reason.

    Returns (result, info). `result` has one row per post, in the input order.
    """
    raw_dir = Path(raw_dir)
    bars = load_bars(raw_dir / config.BARS_FILE)
    history = load_posts(raw_dir / config.POSTS_FILE)[["post_id", "created_at"]]
    info = {"data_version": read_data_version(raw_dir)}
    if model is None:
        model, model_info = load_model()
        info.update(model_info)
    if threshold is None:
        threshold = alert_threshold(model, raw_dir)
    info["alert_threshold"] = threshold

    posts = posts.reset_index(drop=True)
    feats, skipped = serving_features(posts, history, bars, online)
    out = posts[["post_id", "created_at"]].copy()
    out["probability"] = np.nan
    out["alert"] = False
    out["reason"] = ""
    if len(feats):
        x = candidate_input(feats, posts)
        proba = model.predict_proba(x)[:, 1]
        scored = pd.DataFrame({"post_id": feats["post_id"].to_numpy(), "probability": proba})
        out = out.drop(columns="probability").merge(scored, on="post_id", how="left")
        out["alert"] = out["probability"] >= threshold
        feat_cols = feats.set_index("post_id")[["sigma", "last_bar_return", "posts_prev_60min"]]
        out = out.merge(feat_cols, left_on="post_id", right_index=True, how="left")
    for _, row in skipped.iterrows():
        out.loc[out["post_id"] == row["post_id"], "reason"] = row["reason"]
    return out, info
