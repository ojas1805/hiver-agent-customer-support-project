
"""
Build and save normalized MiniLM retrieval embeddings.

Run from repository root:

python src/replies/build_index.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "apple_support_pairs.csv"
)

OUTPUT_PATH = (
    ROOT
    / "models"
    / "retrieval_embeddings.npy"
)

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


def main():
    df = pd.read_csv(DATA_PATH)

    texts = (
        df["customer_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=128,
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    embeddings = (
        embeddings /
        np.clip(norms, 1e-12, None)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_PATH,
        embeddings
    )

    print(
        f"Saved {len(embeddings)} embeddings to "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
