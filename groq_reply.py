
"""
Grounded Groq reply generation.

The model is instructed to use only historical AppleSupport
responses supplied as retrieval evidence.
"""

import os
import re
from groq import Groq


MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


class GroqReplyGenerator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set."
            )

        self.client = Groq(
            api_key=api_key
        )

    @staticmethod
    def _valid_response(text):
        if not text:
            return False

        text = text.strip()

        if len(text) < 20:
            return False

        # Reject obvious incomplete/truncated endings.
        if text.endswith(
            ("...", "and", "or", "to", "a", "an", "the")
        ):
            return False

        return True

    def generate(self, customer_message, retrieved_examples):
        evidence = []

        for i, item in enumerate(
            retrieved_examples,
            start=1
        ):
            evidence.append(
                f"[Historical example {i}]\n"
                f"Customer: {item['customer_text']}\n"
                f"AppleSupport response: "
                f"{item['brand_response']}"
            )

        evidence_text = "\n\n".join(evidence)

        prompt = f"""
You are an AppleSupport customer-support drafting assistant.

Customer message:
{customer_message}

Historical AppleSupport responses:
{evidence_text}

Write a customer-facing reply grounded ONLY in the
historical AppleSupport responses above.

Rules:
- Do not use outside knowledge.
- Do not invent policies, refunds, prices, timelines,
  guarantees, troubleshooting steps, or technical claims.
- Do not claim that a problem is fixed or diagnosed.
- You may ask for information when historical responses
  support asking for that information.
- Keep the response to 1–3 complete sentences.
- Return only the customer-facing reply.
"""

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict grounded support "
                        "response generator."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        text = (
            response.choices[0].message.content or ""
        ).strip()

        if not self._valid_response(text):
            return {
                "reply": "",
                "source": "invalid_groq_response",
            }

        return {
            "reply": text,
            "source": "groq_grounded",
        }
