"""Local embedding model (CPU)."""

from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=len(texts) > 200
        )
        return np.asarray(v, dtype="float32")
