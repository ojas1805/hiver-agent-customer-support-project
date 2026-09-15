"""
Groq-powered grounded response generation for AppleSupport.
"""

import os

from groq import Groq


class GroqReplyGenerator:
    def __init__(
        self,
        model="openai/gpt-oss-120b",
        temperature=0.2,
        max_tokens=180,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Set it before running the support agent."
            )

        self.client = Groq(api_key=api_key)

    def _build_prompt(
        self,
        customer_message,
        retrieval_results,
    ):
        examples = []

        for item in retrieval_results:
            customer_text = str(
                item.get("customer_text", "")
            )
            brand_reply = str(
                item.get("brand_reply", "")
            )

            examples.append(
                f"Customer: {customer_text}\n"
                f"AppleSupport: {brand_reply}"
            )

        historical_context = "\n\n".join(examples)

        prompt = f"""
You are an AppleSupport customer-service assistant.

Customer message:
{customer_message}

Historical AppleSupport examples:
{historical_context}

Write a concise, polite and helpful support response.

Requirements:
- Use the historical examples only as grounding evidence.
- Do not invent Apple policies, refunds, compensation, prices,
  timelines, guarantees, diagnoses, or unsupported technical claims.
- Do not claim that you performed an action you cannot perform.
- Give a practical next step when the historical evidence supports one.
- If the evidence is insufficient, direct the customer toward
  an appropriate official Apple support channel.
- Keep the response concise.
"""

        return prompt.strip()

    def generate(
        self,
        customer_message,
        retrieval_results,
    ):
        prompt = self._build_prompt(
            customer_message,
            retrieval_results,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful AppleSupport assistant. "
                            "Ground responses in the supplied historical "
                            "evidence and avoid unsupported claims."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            reply = response.choices[0].message.content

            if reply is None:
                return {
                    "reply": None,
                    "valid": False,
                    "source": "groq",
                    "reason": "Empty Groq response.",
                }

            reply = reply.strip()

            if not reply:
                return {
                    "reply": None,
                    "valid": False,
                    "source": "groq",
                    "reason": "Empty Groq response.",
                }

            return {
                "reply": reply,
                "valid": True,
                "source": "groq",
                "reason": None,
            }

        except Exception as exc:
            return {
                "reply": None,
                "valid": False,
                "source": "groq_error",
                "reason": str(exc),
            }