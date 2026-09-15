# Large Model Artifacts

The complete trained project contains three large artifacts:

- models/distilbert_intent_classifier/
- models/retrieval_embeddings.npy
- data/processed/apple_support_pairs.csv

These are excluded from the GitHub-friendly source package because of their size.

The complete archive contains them.

The DistilBERT model belongs at:
models/distilbert_intent_classifier/

The retrieval matrix belongs at:
models/retrieval_embeddings.npy

Expected retrieval shape: (104409, 384)

The historical pair dataset belongs at:
data/processed/apple_support_pairs.csv

Retrieval embeddings can be rebuilt with:
python src/replies/build_index.py
