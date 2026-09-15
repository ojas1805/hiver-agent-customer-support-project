
"""
End-to-end AppleSupport Hiver agent.

Flow:
customer message
    -> intent classification
    -> historical retrieval
    -> grounded Groq response
    -> escalation decision
"""

from pathlib import Path

from classifier import IntentClassifier
from retrieval import HistoricalRetriever
from groq_reply import GroqReplyGenerator
from policy import decide_escalation


class AppleSupportAgent:
    def __init__(self):
        self.classifier = IntentClassifier()
        self.retriever = HistoricalRetriever()
        self.reply_generator = GroqReplyGenerator()

    def run(self, customer_message):
        # ----------------------------------------------------
        # Intent
        # ----------------------------------------------------
        intent_result = self.classifier.predict(
            customer_message
        )

        intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        # ----------------------------------------------------
        # Historical retrieval
        # ----------------------------------------------------
        retrieval_results = self.retriever.retrieve(
            customer_message,
            top_k=3
        )

        if retrieval_results:
            top_strength = retrieval_results[0]["strength"]
            top_score = retrieval_results[0]["score"]
        else:
            top_strength = "WEAK"
            top_score = 0.0

        # ----------------------------------------------------
        # Generate grounded response
        # ----------------------------------------------------
        reply_result = self.reply_generator.generate(
            customer_message,
            retrieval_results
        )

        reply = reply_result["reply"]
        response_source = reply_result["source"]

        # ----------------------------------------------------
        # Escalation
        # ----------------------------------------------------
        escalation = decide_escalation(
            intent=intent,
            confidence=confidence,
            retrieval_strength=top_strength,
            response_source=response_source,
        )

        return {
            "customer_message": customer_message,
            "intent": intent,
            "intent_confidence": confidence,
            "intent_source": intent_result["source"],
            "retrieval_top_score": top_score,
            "retrieval_strength": top_strength,
            "reply": reply,
            "response_source": response_source,
            "escalate": escalation["escalate"],
            "escalation_reason": escalation["reason"],
            "retrieved_examples": retrieval_results,
        }


if __name__ == "__main__":
    agent = AppleSupportAgent()

    message = input(
        "Enter customer message: "
    ).strip()

    result = agent.run(message)

    print("\n" + "=" * 60)
    print("APPLE SUPPORT AGENT")
    print("=" * 60)

    print("Intent:", result["intent"])
    print(
        "Confidence:",
        round(result["intent_confidence"], 4)
    )

    print(
        "Retrieval score:",
        round(result["retrieval_top_score"], 4)
    )

    print(
        "Retrieval strength:",
        result["retrieval_strength"]
    )

    print("\nDraft reply:")
    print(result["reply"])

    print("\nEscalate:", result["escalate"])
    print(
        "Reason:",
        result["escalation_reason"]
    )
