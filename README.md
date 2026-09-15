# Hiver AI Support Agent — AppleSupport

AI support agent built for the Hiver take-home assignment using the Customer Support on Twitter dataset.

## Overview

The agent:
1. Classifies customer messages into 12 support intents.
2. Retrieves similar historical AppleSupport conversations.
3. Generates a grounded reply using Groq.
4. Decides whether to auto-handle or escalate.

## Architecture

Customer Message
→ DistilBERT + rules
→ MiniLM historical retrieval
→ Grounded Groq response
→ Escalation policy
→ Auto-handle / Escalate

## Intent Taxonomy

- ios_update_issue
- app_crash_or_freeze
- device_performance
- battery_issue
- connectivity_issue
- media_playback_issue
- authentication_or_code
- hardware_repair_or_damage
- system_ui_keyboard_or_notification_bug
- icloud_photos_or_mail
- messaging_or_communication
- information_or_how_to

Definitions are in src/intents/intent_definitions.py.

## Dataset

Brand: AppleSupport

AppleSupport rows: 106,860
Usable customer-to-brand pairs: 104,409
Strong silver examples: 3,415
Clean evaluation examples: 110

## Classification

Primary classifier: DistilBERT.

Silver validation:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 90.63% | 90.06% | 90.70% |
| DistilBERT | 94.29% | 93.67% | 94.25% |

Leakage-free development evaluation:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Zero-Shot Groq | 77.27% | 71.30% | 77.78% |
| DistilBERT + rules | 90.91% | 90.50% | 90.50% |

Accuracy improvement over zero-shot baseline: 13.64 percentage points.

Note: the clean evaluation is a semantic development evaluation, not a fully human-labeled benchmark.

## Retrieval

Model: sentence-transformers/all-MiniLM-L6-v2

Top 3 historical examples are retrieved using normalized cosine similarity.

Thresholds:
- STRONG >= 0.60
- MEDIUM >= 0.45
- WEAK < 0.45

Average top-1 similarity: 0.8146.

This represents similarity, not retrieval accuracy.

## Grounded Response Generation

Provider: Groq
Model: openai/gpt-oss-120b

The generator is instructed to use only historical AppleSupport evidence and avoid unsupported policies, refunds, prices, timelines, guarantees, diagnoses, and troubleshooting claims.

Empty or incomplete responses are rejected.

## Response Evaluation

20 representative cases were tested.

Valid responses: 6
Invalid or empty responses: 14
Validity: 30%

LLM-as-judge on the 6 valid responses:

| Dimension | Average |
|---|---:|
| Groundedness | 5.00 / 5 |
| Helpfulness | 4.67 / 5 |
| Tone | 5.00 / 5 |
| Safety | 5.00 / 5 |
| Overall | 5.00 / 5 |

The sample is small.

### Important Metric Distinction

The 30% response-validity result and the 79.09% auto-handle result come from different evaluations.

- Response validity was measured on a separate 20-case sample: 6/20 valid.
- Auto-handle was measured by the escalation policy on the 110-case clean development evaluation: 87/110.

Therefore, the 79.09% auto-handle rate should not be interpreted as a response-validity rate, and the 30% response-validity result should not be interpreted as the validity rate of the 110-case escalation evaluation.

The response guard rejects incomplete or unsafe output instead of sending it to the customer.

## Escalation

The final policy considers:
- low intent confidence
- weak historical evidence
- authentication/account issues
- invalid or unavailable responses
- conservative fallback behavior

Final system:
- Auto-handle: 87 / 110 (79.09%)
- Escalate: 23 / 110 (20.91%)

## Required Baseline

Zero-shot Groq baseline:
- Accuracy: 77.27%
- Macro F1: 71.30%
- Weighted F1: 77.78%

Baseline escalation:
- Auto-handle: 96 / 110 (87.27%)
- Escalate: 14 / 110 (12.73%)

## Golden Set

The assignment requests 150–250 hand-labeled examples.

Current human-labeled examples: 100.

The current 100-example set is below the original 150–250 target.

## Known Limitations

1. The golden set is below the requested 150–250 examples.
2. The clean evaluation is semantic rather than fully human labeled.
3. information_or_how_to has zero examples in the clean evaluation.
4. Response validity was only 30% on the 20-case sample.
5. LLM-as-judge used only 6 valid responses.
6. Retrieval is currently implemented with MiniLM embeddings and NumPy similarity.
7. A separate optional Groq risk-check pass was not included.

## Running

Install:

pip install -r requirements.txt

Set the Groq key:

export GROQ_API_KEY='YOUR_KEY'

Build embeddings:

python src/replies/build_index.py

Run:

python pipeline.py

## Human vs LLM Judge Agreement

Six valid generated responses were independently evaluated by a human reviewer using the same 1–5 rubric used by the LLM judge.

Across all valid paired dimension ratings:

- Exact agreement: **100%**
- Mean absolute error: **0.000**
- Cohen's kappa: **1.000**
- Quadratic weighted kappa: **1.000**

Per-dimension exact agreement was 100% for groundedness, helpfulness, tone, safety, and overall rating.

The sample contains only six responses, so this is evidence of agreement on the evaluated sample rather than a broad reliability estimate.

Detailed paired results are stored in:

`data/processed/human_vs_llm_paired_results.csv`

`data/processed/judge_human_agreement.csv`

`data/processed/judge_human_agreement_summary.csv`

## Quick Start

There are two supported reproduction paths.

### Path A — Quick Demo

This path uses the supplied trained artifacts and is intended for fast repository verification.

#### 1. Install dependencies

```bash
pip install -r requirements.txt
