
"""
Evaluate the saved DistilBERT intent classifier.

Run from repository root:

python src/eval/evaluate_intent.py
"""

from pathlib import Path
import pandas as pd

from src.intents.classifier import IntentClassifier
from src.eval.metrics import classification_metrics


ROOT = Path(__file__).resolve().parents[2]

EVAL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "clean_eval_dataset.csv"
)


def main():

    df = pd.read_csv(EVAL_PATH)

    classifier = IntentClassifier()

    predictions = []

    for _, row in df.iterrows():

        result = classifier.predict(
            row["customer_text"]
        )

        predictions.append({
            "customer_text":
                row["customer_text"],
            "expected_intent":
                row["intent"],
            "predicted_intent":
                result["intent"],
            "confidence":
                result["confidence"],
            "source":
                result["source"],
        })

    results = pd.DataFrame(predictions)

    valid = results[
        results["predicted_intent"].notna()
    ]

    metrics = classification_metrics(
        valid["expected_intent"],
        valid["predicted_intent"],
    )

    print("=" * 60)
    print("DISTILBERT INTENT EVALUATION")
    print("=" * 60)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    output_path = (
        ROOT
        / "data"
        / "processed"
        / "reproducible_intent_evaluation.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()
