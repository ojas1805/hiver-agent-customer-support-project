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
        # Repository root = directory containing this file.
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

        # The actual CSV column is brand_response.
        if "brand_response" not in self.df.columns:
            raise ValueError(
                "apple_support_pairs.csv must contain brand_response"
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
            texts = (
                self.df["customer_text"]
                .fillna("")
                .astype(str)
                .tolist()
            )

            self.embeddings = self.embedder.encode(
                texts,
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
            convert_to_numpy=True,
        )[0]

        norm = np.linalg.norm(query_embedding)
        query_embedding = (
            query_embedding
            / max(norm, 1e-12)
        )

        # Because both vectors are normalized,
        # dot product is cosine similarity.
        scores = self.embeddings @ query_embedding

        top_k = min(top_k, len(scores))
        indices = np.argsort(scores)[-top_k:][::-1]

        results = []

        for idx in indices:
            score = float(scores[idx])

            results.append(
                {
                    "customer_text": self.df.iloc[idx]["customer_text"],
                    "brand_reply": self.df.iloc[idx]["brand_response"],
                    "score": score,
                    "strength": self._strength(score),
                }
            )

        return results