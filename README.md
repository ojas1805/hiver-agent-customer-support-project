# Hiver AI Support Agent — AppleSupport

An end-to-end AI customer-support agent built for the Hiver take-home assignment using the **Customer Support on Twitter** dataset.

The system is designed around one principle: **prefer grounded, safe support over unsupported AI-generated answers**.

---

## 🚀 Overview

For every incoming customer message, the agent:

1. Classifies the customer's issue into one of 12 support intents.
2. Retrieves similar historical AppleSupport conversations.
3. Generates a customer-facing response grounded in those historical responses.
4. Decides whether the request can be auto-handled or should be escalated.

### Architecture

```text
Customer Message
       |
       v
+-----------------------------+
| DistilBERT Intent Classifier|
| + High-Precision Rules      |
+-----------------------------+
       |
       v
+-----------------------------+
| Historical Retrieval        |
| MiniLM Embeddings           |
| Top-3 Similar Conversations |
+-----------------------------+
       |
       v
+-----------------------------+
| Grounded Groq Response      |
| Generation                  |
+-----------------------------+
       |
       v
+-----------------------------+
| Escalation Policy           |
| Confidence + Evidence       |
| + Response Safety Checks    |
+-----------------------------+
       |
       v
 Auto-handle / Escalate
