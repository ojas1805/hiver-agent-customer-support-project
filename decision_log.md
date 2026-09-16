# Decision Log

## Brand selection
AppleSupport was selected because it provided 104,409 usable customer-to-brand response pairs.
This was more suitable for historical grounding than the initially considered British Airways slice.

## Intent taxonomy
A 12-intent taxonomy was chosen to distinguish common Apple support issue categories while remaining manageable for classification.

## Silver labeling
MiniLM semantic similarity was used to construct a strong silver dataset because the source dataset does not provide support-intent labels.
The resulting strong silver dataset contains 3,415 examples.
These are silver labels and are not human ground truth.

## Classifier
DistilBERT was selected as the main learned classifier after outperforming TF-IDF + Logistic Regression on silver validation.

## Rule overrides
High-precision rules were added for obvious battery, connectivity, crash, authentication, messaging, media, and hardware signals.

## Retrieval
MiniLM embeddings with cosine similarity and top-3 retrieval were selected to ground replies in historical AppleSupport behavior.

## LLM
Groq was used for LLM calls with openai/gpt-oss-120b.

## Grounding
Response generation is restricted to retrieved historical AppleSupport evidence to reduce unsupported claims.

## Response rejection
Empty, very short, or incomplete generated responses are rejected and can trigger escalation.

## Authentication
Authentication/account-related issues are escalated conservatively.

## Evaluation leakage
Earlier evaluation that permitted self-retrieval was discarded. The final clean evaluation excludes silver-training customer texts and exact self matches.

## Golden set
The human-checked evaluation set was expanded to 150 examples, meeting the lower bound of the requested 150–250-example requirement.

## Optional risk checker
A separate final Groq risk-check pass was not included. Safety currently relies on grounded prompting, output validation, and deterministic escalation.