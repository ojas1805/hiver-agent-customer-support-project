
"""
Escalation policy for the final support agent.
"""


def decide_escalation(
    intent,
    confidence,
    retrieval_strength,
    response_source,
):
    reasons = []

    # Low classifier confidence
    if confidence is None or confidence < 0.65:
        reasons.append(
            "low intent confidence"
        )

    # Weak retrieval
    if retrieval_strength == "WEAK":
        reasons.append(
            "weak historical evidence"
        )

    # Authentication/account issues need human review
    if intent == "authentication_or_code":
        reasons.append(
            "authentication/account caution"
        )

    # Historical fallback is deliberately conservative.
    if response_source == "historical_fallback":
        reasons.append(
            "historical fallback requires human review"
        )

    # Empty/invalid response
    if response_source in (
        "invalid_groq_response",
        "no_response",
    ):
        reasons.append(
            "no safe generated response"
        )

    if reasons:
        return {
            "escalate": True,
            "reason": "; ".join(reasons),
        }

    return {
        "escalate": False,
        "reason": "all safety and confidence checks passed",
    }
