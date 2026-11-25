"""Write the small seed that lets a clean clone build the whole system without DVC or MinIO.

    uv run python scripts/export_seed.py

deploy/seed/train_table.parquet  the training table of Lab 2 (text, features, label, split) for
                                 every usable post: the model of the registry bootstrap is trained
                                 on its train part, the drift reference window is a sample of it,
                                 and the replay script draws held-out rows from its test part;
deploy/seed/corpus.parquet       the retrieval corpus of Lab 3, indexed by the rag-ingest job;
deploy/seed/DATA_VERSION         the DVC hash of data/raw the two files were derived from.
Both files are derived from public posts; the source and its licence are stated in the README.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops_labs import config, dataset
from mlops_labs.data import load_bars, load_posts
from mlops_labs.env import ROOT
from mlops_labs.labels import build_windows, label_posts

OUT = ROOT / "deploy" / "seed"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "data" / "raw"
    table, data_version = dataset.from_raw(raw)
    table["split"] = dataset.split_name(table["session_date"])
    table.to_parquet(OUT / "train_table.parquet", index=False)
    (OUT / "DATA_VERSION").write_text(data_version + "\n", encoding="utf-8")

    posts = load_posts(raw / config.POSTS_FILE)
    labelled = label_posts(posts, build_windows(load_bars(raw / config.BARS_FILE)))
    keep = labelled[~labelled["text_empty"] & ~labelled["dup_text"]]
    corpus = keep[["post_id", "created_at", "text", "y", "r"]].rename(columns={"r": "ret"})
    corpus.to_parquet(OUT / "corpus.parquet", index=False)
    print(f"table {len(table)} rows {table['split'].value_counts().to_dict()}, "
          f"corpus {len(corpus)} posts, data {data_version}")


if __name__ == "__main__":
    main()
