# Large Model Artifacts

The GitHub source repository intentionally excludes the large trained artifacts.

The complete artifact bundle contains:

- `models/distilbert_intent_classifier/`
- `models/retrieval_embeddings.npy`
- `data/processed/apple_support_pairs.csv`

These files are excluded because of their size.

## Quick Demo

For the fastest way to run the existing agent, place the supplied artifact bundle into the repository so that the paths above exist.

Then run:

```bash
pip install -r requirements.txt
export GROQ_API_KEY='YOUR_KEY'
python demo.py
