
"""
Evaluation helpers.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


def classification_metrics(
    y_true,
    y_pred,
):
    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def save_classification_report(
    y_true,
    y_pred,
    path,
):
    report = classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
