
"""
Data preparation utilities for AppleSupport historical pairs.
"""

from pathlib import Path
import re
import pandas as pd


def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_pairs(path=None):
    if path is None:
        root = Path(__file__).resolve().parents[2]
        path = (
            root
            / "data"
            / "processed"
            / "apple_support_pairs.csv"
        )

    df = pd.read_csv(path)

    df["customer_text_clean"] = (
        df["customer_text"]
        .fillna("")
        .map(clean_text)
    )

    df["brand_response_clean"] = (
        df["brand_response"]
        .fillna("")
        .map(clean_text)
    )

    return df


def basic_summary(df):
    return {
        "rows": len(df),
        "unique_customer_messages":
            df["customer_text"].nunique(),
        "unique_brand_responses":
            df["brand_response"].nunique(),
    }
