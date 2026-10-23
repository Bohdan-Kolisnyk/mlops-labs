"""Numbers of docs/design.md, reproduced from the real files. Skipped without the data
(`dvc pull` first)."""

from pathlib import Path

import pytest

from mlops_labs import config
from mlops_labs.data import load_bars, load_posts
from mlops_labs.labels import build_windows, label_posts

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
pytestmark = pytest.mark.skipif(
    not (RAW / config.POSTS_FILE).exists(), reason="data/raw is not pulled"
)


def test_design_numbers():
    posts = load_posts(RAW / config.POSTS_FILE)
    labelled = label_posts(posts, build_windows(load_bars(RAW / config.BARS_FILE)))
    assert len(posts) == 36_924
    assert len(labelled) == 4_877
    assert labelled["session_date"].nunique() == 621
    assert int(labelled["text_empty"].sum()) == 1_136
    assert int(labelled["dup_text"].sum()) == 118
    d = labelled["session_date"]
    train = labelled[d <= config.TRAIN_END]
    val = labelled[(d >= config.VAL_START) & (d <= config.VAL_END)]
    test = labelled[d >= config.TEST_START]
    assert (len(train), int(train["y"].sum())) == (3_056, 442)
    assert (len(val), int(val["y"].sum())) == (613, 98)
    assert (len(test), int(test["y"].sum())) == (1_208, 164)
