# Baseline Evaluation

## Required Zero-Shot Groq Baseline

The assignment-required baseline uses:

1. Zero-shot Groq intent classification.
2. Groq response generation without historical retrieval.
3. Simple rule-based escalation.

### Intent classification

| Metric | Result |
|---|---:|
| Accuracy | 77.27% |
| Macro F1 | 71.30% |
| Weighted F1 | 77.78% |
| Successful calls | 110 / 110 |
| Misclassified | 25 / 110 |

### Escalation

| Decision | Count | Rate |
|---|---:|---:|
| Auto-handle | 96 | 87.27% |
| Escalate | 14 | 12.73% |

Escalation reasons:
- Authentication/account caution: 13
- Low intent confidence: 1

## Comparison

| System | Accuracy | Macro F1 |
|---|---:|---:|
| Zero-Shot Groq | 77.27% | 71.30% |
| DistilBERT + rules | 90.91% | 90.50% |

Accuracy gain:
13.64 percentage points.

Macro F1 gain:
19.20 percentage points.

## Interpretation

The domain-trained classifier provides better intent routing than the zero-shot baseline on the clean development evaluation.

The final system is also deliberately more conservative in escalation because it incorporates additional checks beyond the baseline confidence rule.
