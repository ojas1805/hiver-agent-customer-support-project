
"""
Evaluate the end-to-end support agent.

Run from repository root:

python src/eval/evaluate_agent.py

Note:
This evaluation invokes Groq and therefore requires
GROQ_API_KEY.
"""

from pathlib import Path
import pandas as pd

from pipeline import AppleSupportAgent


ROOT = Path(__file__).resolve().parents[2]

EVAL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "clean_eval_dataset.csv"
)


def main():

    df = pd.read_csv(EVAL_PATH)

    agent = AppleSupportAgent()

    outputs = []

    for i, row in df.iterrows():

        print(
            f"Processing {i + 1}/{len(df)}..."
        )

        try:

            result = agent.run(
                row["customer_text"]
            )

            outputs.append({
                "customer_text":
                    row["customer_text"],
                "expected_intent":
                    row["intent"],
                "predicted_intent":
                    result["intent"],
                "intent_confidence":
                    result["intent_confidence"],
                "retrieval_top_score":
                    result["retrieval_top_score"],
                "retrieval_strength":
                    result["retrieval_strength"],
                "reply":
                    result["reply"],
                "response_source":
                    result["response_source"],
                "escalate":
                    result["escalate"],
                "escalation_reason":
                    result["escalation_reason"],
            })

        except Exception as exc:

            outputs.append({
                "customer_text":
                    row["customer_text"],
                "expected_intent":
                    row["intent"],
                "error":
                    str(exc),
            })

    output_path = (
        ROOT
        / "data"
        / "processed"
        / "reproducible_agent_evaluation.csv"
    )

    pd.DataFrame(outputs).to_csv(
        output_path,
        index=False
    )

    print("\n✅ Evaluation complete")
    print("Saved:")
    print(output_path)


if __name__ == "__main__":
    main()
