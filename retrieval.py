"""
Historical AppleSupport retrieval using MiniLM embeddings
and cosine similarity implemented as normalized dot product.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class HistoricalRetriever:
    def __init__(
        self,
        data_path=None,
        embedding_path=None,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ):
        # The repository uses a flat structure, so the repository root
        # is the directory containing this retrieval.py file.
        root = Path(__file__).resolve().parent

        # Historical customer/support pairs
        if data_path is None:
            data_path = (
                root
                / "data"
                / "processed"
                / "apple_support_pairs.csv"
            )

        # Precomputed MiniLM embeddings
        if embedding_path is None:
            embedding_path = (
                root
                / "models"
                / "retrieval_embeddings.npy"
            )

        self.data_path = Path(data_path)
        self.embedding_path = Path(embedding_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Historical pairs not found: {self.data_path}"
            )

        self.df = pd.read_csv(self.data_path)

        if "customer_text" not in self.df.columns:
            raise ValueError(
                "apple_support_pairs.csv must contain customer_text"
            )

        self.embedder = SentenceTransformer(model_name)

        # Load precomputed embeddings when available.
        if self.embedding_path.exists():
            self.embeddings = np.load(self.embedding_path)

            if len(self.embeddings) != len(self.df):
                raise ValueError(
                    "Embedding count does not match historical rows."
                )

            # Make sure stored embeddings are normalized.
            norms = np.linalg.norm(
                self.embeddings,
                axis=1,
                keepdims=True,
            )

            self.embeddings = (
                self.embeddings
                / np.clip(norms, 1e-12, None)
            )

        else:
            # Fallback: generate embeddings locally.
            self.embeddings = self.embedder.encode(
                self.df["customer_text"]
                .fillna("")
                .tolist(),
                convert_to_numpy=True,
                show_progress_bar=True,
            )

            norms = np.linalg.norm(
                self.embeddings,
                axis=1,
                keepdims=True,
            )

            self.embeddings = (
                self.embeddings
                / np.clip(norms, 1e-12, None)
            )

    def _strength(self, score):
        if score >= 0.60:
            return "STRONG"
        elif score >= 0.45:
            return "MEDIUM"
        return "WEAK"

   def retrieve(self, query, top_k=3):
    query_embedding = self.embedder.encode(
        [str(query)],
        convert_to_numpy=True
    )[0]

    norm = np.linalg.norm(query_embedding)

    query_embedding = (
        query_embedding /
        max(norm, 1e-12)
    )
        # Because both vectors are normalized,
        # dot product is cosine similarity.
        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_indices:
            row = self.df.iloc[int(idx)]
            score = float(scores[idx])

            results.append(
                {
                    "customer_text": str(
                        row.get("customer_text", "")
                    ),
                    "brand_response": str(
                        row.get("brand_response", "")
                    ),
                    "score": score,
                    "strength": self._strength(score),
                }
            )

        return results