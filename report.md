# Hiver AI Support Agent — Technical Report

## 1. Problem Framing

The goal is to build a trustworthy AI customer-support agent for AppleSupport.

For each incoming message, good performance means:
- useful intent routing;
- responses grounded in historical AppleSupport behavior;
- avoidance of unsupported claims;
- escalation when uncertainty or risk is too high.

The system does not attempt autonomous account access, identity verification, refunds, payment execution, or unsupported technical diagnosis.

## 2. Data and Brand Choice

The source dataset is Customer Support on Twitter.

AppleSupport was selected because the dataset contained 106,860 AppleSupport rows and 104,409 usable customer-to-AppleSupport pairs.

A strong silver set of 3,415 examples was constructed for classifier training.

The clean development evaluation contains 110 examples.

## 3. System Architecture

```text
Customer Message
      |
      v
DistilBERT Intent Classifier + Rules
      |
      v
MiniLM Historical Retrieval
      |
      v
Grounded Groq Response Generation
      |
      v
Confidence / Evidence / Safety Checks
      |
      v
Auto-handle OR Escalate
```

## 4. Intent Classification

The system uses 12 support intents.

Training labels were created using MiniLM semantic similarity to intent seed examples. These are silver labels rather than human ground truth.

The primary classifier is DistilBERT.

High-precision rules handle obvious battery, connectivity, crash, authentication, messaging, media, and hardware cases.

## 5. Results vs Baselines

| System | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority-class trivial baseline | 9.09% | 0.0139 | 0.0152 |
| TF-IDF + Logistic Regression | 90.63% | 90.06% | 90.70% |
| Zero-Shot Groq | 77.27% | 71.30% | 77.78% |
| DistilBERT + rules | 90.91% | 90.50% | 90.50% |

The trivial baseline predicts authentication_or_code for every evaluation case.

TF-IDF + Logistic Regression provides a simple learned baseline.

Zero-Shot Groq provides the required LLM baseline without domain training or historical retrieval.

On the clean development evaluation, DistilBERT + rules improved accuracy over Zero-Shot Groq by 13.64 percentage points.

DistilBERT silver validation achieved 94.29% accuracy and 93.67% macro F1.

## 6. Historical Retrieval and Grounding

Historical retrieval uses sentence-transformers/all-MiniLM-L6-v2.

Three similar historical customer-to-AppleSupport conversations are retrieved using normalized cosine similarity.

Average top-1 similarity was 0.8146.

Similarity is not retrieval accuracy.

The response generator receives historical evidence and is instructed not to invent policies, refunds, prices, timelines, guarantees, diagnoses, or unsupported troubleshooting steps.

## 7. Escalation

The final policy considers classifier confidence, retrieval strength, authentication/account risk, and response validity.

Final clean evaluation:
- Auto-handle: 87 / 110 (79.09%)
- Escalate: 23 / 110 (20.91%)

Zero-Shot baseline:
- Auto-handle: 96 / 110 (87.27%)
- Escalate: 14 / 110 (12.73%)

The final system is intentionally more conservative.

## 8. Response Quality Evaluation

Twenty representative cases were evaluated.

- Valid responses: 6 / 20
- Invalid or empty responses: 14 / 20
- Validity: 30%

The guard rejects incomplete or unsafe output instead of sending it to the customer.

The six valid responses were evaluated with an LLM-as-judge:

| Dimension | Average |
|---|---:|
| Groundedness | 5.00 / 5 |
| Helpfulness | 4.67 / 5 |
| Tone | 5.00 / 5 |
| Safety | 5.00 / 5 |
| Overall | 5.00 / 5 |

## 9. Human-vs-LLM Judge Agreement

The same six responses were independently scored by a human reviewer using the same 1–5 rubric.

| Dimension | Valid pairs | Exact agreement | MAE | Quadratic weighted kappa |
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

The N/A kappa values occur where both raters assigned the same constant score for every valid observation, making kappa mathematically undefined.

This is evidence of agreement on this six-response sample, but the sample is too small to establish broad judge reliability.

## 10. Golden Evaluation Set

The current development evaluation should be interpreted separately from a full human golden benchmark.

The larger clean evaluation is a semantic development set rather than a fully human-labelled benchmark.

## 11. Top 5 Failure Modes

### 1. iOS update vs device performance
Tweets can describe performance degradation immediately after an update.

### 2. iOS update vs system UI
Update complaints can specifically describe keyboard or interface behavior.

### 3. iCloud/Mail vs authentication
Mail or iCloud access problems may contain authentication vocabulary.

### 4. Authentication vs messaging
Verification or account language can overlap with communication problems.

### 5. Response-generation reliability
Groq occasionally returned empty or incomplete responses. These are rejected and escalated rather than exposed to customers.

## 12. What Is Misleading About My Headline Number?

The headline classification number is 90.91% accuracy.

That number is useful but potentially misleading.

First, the 110-example clean evaluation is a semantic development set rather than a fully human-labelled benchmark.

Second, information_or_how_to has zero examples in the evaluation sample, so the evaluation does not provide evidence for that class.

Third, the silver training labels were automatically generated using semantic similarity and are not human ground truth.

Fourth, retrieval similarity does not guarantee that a retrieved historical answer is correct or sufficient.

Fifth, response generation had only 30% validity on the sampled 20 cases.

Therefore 90.91% should be interpreted as a leakage-free development classification result, not production end-to-end support accuracy.

## 13. What I Would Do With One More Week

1. Expand human evaluation coverage.
2. Add more hard-negative intent examples.
3. Improve separation of update/performance/UI cases.
4. Improve Groq response reliability.
5. Add the optional response-risk checker.
6. Persist retrieval with FAISS and benchmark latency.
7. Expand response-quality evaluation beyond six judged responses.

## 14. Conclusion

The final architecture combines domain-specific classification, historical retrieval, grounded generation, and conservative escalation.

The clean development evaluation shows 90.91% accuracy versus 77.27% for Zero-Shot Groq and 9.09% for the majority baseline.

The response evaluator also demonstrates perfect agreement between the human reviewer and LLM judge on the six paired responses tested, although the sample is small.
