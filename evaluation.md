# Evaluation

## Dataset

Historical AppleSupport pairs: 104,409
Strong silver examples: 3,415
Clean leakage-free development evaluation: 110

## Classification

| System | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority-class baseline | 9.09% | 0.0139 | 0.0152 |
| TF-IDF + Logistic Regression | 90.63% | 90.06% | 90.70% |
| Zero-Shot Groq | 77.27% | 71.30% | 77.78% |
| DistilBERT + rules | 90.91% | 90.50% | 90.50% |

DistilBERT silver validation:
- Accuracy: 94.29%
- Macro F1: 93.67%
- Weighted F1: 94.25%

## Retrieval

Average top-1 MiniLM similarity: 0.8146.

All 110 clean evaluation examples crossed the STRONG threshold after exact self-match exclusion.

This is similarity, not retrieval accuracy.

## Response evaluation

20 representative cases were tested:

- Valid responses: 6
- Invalid or empty: 14
- Validity: 30%

## LLM-as-judge

Six valid responses were evaluated:

- Groundedness: 5.00 / 5
- Helpfulness: 4.67 / 5
- Tone: 5.00 / 5
- Safety: 5.00 / 5
- Overall: 5.00 / 5

## Human-vs-LLM judge agreement

The same six responses were independently scored by a human using the same rubric.

| Dimension | n | Exact agreement | MAE | Quadratic weighted kappa |
|---|---:|---:|---:|---:|
| Groundedness | 6 | 100% | 0.000 | N/A |
| Helpfulness | 6 | 100% | 0.000 | 1.000 |
| Tone | 6 | 100% | 0.000 | N/A |
| Safety | 5 | 100% | 0.000 | N/A |
| Overall | 6 | 100% | 0.000 | N/A |

Across all valid dimension ratings:

- Exact agreement: 100%
- Mean absolute error: 0.000
- Cohen's kappa: 1.000
- Quadratic weighted kappa: 1.000

Kappa is mathematically undefined for dimensions where both raters used exactly the same constant rating for every observation.

The agreement result is based on only six responses and therefore should not be generalized to all support replies.

## Escalation

Final system:
- Auto-handle: 87 / 110
- Escalate: 23 / 110

Zero-Shot baseline:
- Auto-handle: 96 / 110
- Escalate: 14 / 110

## Limitations

The clean evaluation is semantic rather than fully human labeled.

information_or_how_to has no examples in the clean evaluation.

Response generation validity was only 30% on the 20-case sample.

Human-vs-LLM agreement is based on only six judged responses.
