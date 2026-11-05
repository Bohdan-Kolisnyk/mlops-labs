"""Score a post with the champion model, from raw inputs to a probability.

    uv run python scripts/predict.py --text "Tariffs on China start tomorrow" \\
        --time 2025-03-12T15:05:00Z
    uv run python scripts/predict.py --post-id 114000000000000000      # replay an archive post
    uv run python scripts/predict.py --check 20                        # serving == training?

What happens: the post is placed in its hourly bar and slot, sigma comes from the feature
store (Feast), the other features are built by the training code, the champion model is loaded
from the registry by alias, and the probability is compared with the alert threshold (the top
20% of the validation scores).

--online reads the newest sigma from the online store instead of the point-in-time value.
Needs the stack running (MLflow) and `feast apply` + `demo.py` done once in feature_repo/.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from mlops_labs import config, dataset, models, predict
from mlops_labs.data import load_posts
from mlops_labs.env import ROOT, load_env

RAW = ROOT / "data" / "raw"


def show(result: pd.DataFrame, info: dict) -> None:
    print(
        f"model {predict.MODEL_NAME} version {info.get('version', '?')} "
        f"(data {info['data_version']}, commit {str(info.get('git_commit', '?'))[:10]})"
    )
    print(f"alert threshold (top 20% of validation scores): {info['alert_threshold']:.4f}\n")
    for _, r in result.iterrows():
        if r["reason"]:
            print(f"post {r['post_id']} at {r['created_at']}: not scored, {r['reason']}")
            continue
        print(f"post {r['post_id']} at {r['created_at']}")
        print(
            f"  sigma {r['sigma']:.6f}, last bar return {r['last_bar_return']:+.5f}, "
            f"posts in the previous 60 minutes {int(r['posts_prev_60min'])}"
        )
        print(f"  probability of an abnormal move: {r['probability']:.4f}")
        print(f"  ALERT: {'YES' if r['alert'] else 'no'}")


def check(n: int, model, info: dict, threshold: float) -> None:
    """Serving path vs training table, for n random test posts: the scores must be equal."""
    table, _ = dataset.from_raw(RAW)
    test = table[dataset.split_name(table["session_date"]) == "test"]
    sample = test.sample(min(n, len(test)), random_state=0)
    expected = model.predict_proba(models.prepare_candidate_input(sample))[:, 1]
    posts = load_posts(RAW / config.POSTS_FILE)
    asked = posts[posts["post_id"].isin(sample["post_id"])].set_index("post_id")
    asked = asked.loc[sample["post_id"]].reset_index()
    result, _ = predict.score_posts(asked, RAW, model=model, threshold=threshold)
    diff = (result["probability"].to_numpy() - expected).__abs__()
    print(f"{len(result)} test posts, scored from text and time only")
    print(f"not scored: {int((result['reason'] != '').sum())}")
    print(f"max |serving - training| probability = {np.nanmax(diff):.2e}")
    assert result["reason"].eq("").all() and np.nanmax(diff) < 1e-9
    print("OK: the serving path gives the same score as the training table")


def main() -> None:
    load_env()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--text", help="text of the post")
    p.add_argument("--time", help="time of the post, UTC, e.g. 2025-03-12T15:05:00Z")
    p.add_argument("--has-media", action="store_true")
    p.add_argument("--post-id", help="replay a post of the archive by its id")
    p.add_argument("--check", type=int, metavar="N", help="compare serving and training on N posts")
    p.add_argument("--online", action="store_true", help="sigma from the online store")
    p.add_argument("--model-uri", default=predict.MODEL_URI)
    args = p.parse_args()

    model, info = predict.load_model(args.model_uri)
    threshold = predict.alert_threshold(model, RAW)

    if args.check:
        check(args.check, model, info, threshold)
        return

    if args.post_id:
        posts = load_posts(RAW / config.POSTS_FILE)
        posts = posts[posts["post_id"] == args.post_id]
        if posts.empty:
            raise SystemExit(f"post {args.post_id} is not in the archive")
        posts = posts[["post_id", "created_at", "text", "has_media"]]
    elif args.text and args.time:
        posts = pd.DataFrame(
            {
                "post_id": ["new"],
                "created_at": [
                    pd.Timestamp(args.time, tz="UTC")
                    if "Z" not in args.time and "+" not in args.time
                    else pd.Timestamp(args.time)
                ],
                "text": [args.text],
                "has_media": [args.has_media],
            }
        )
        posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
    else:
        raise SystemExit("give --text and --time, or --post-id, or --check N")

    result, info2 = predict.score_posts(
        posts, RAW, model=model, threshold=threshold, online=args.online
    )
    show(result, {**info, **info2})


if __name__ == "__main__":
    main()
