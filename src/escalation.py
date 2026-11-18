"""Confidence evaluation and human-escalation decisions.

Confidence is computed from observable checks — classification certainty,
retrieval strength, tool success, volume of evidence, validation result and
customer context — not from a number the language model is asked to invent.
"""

from __future__ import annotations

from config import Settings

WEIGHTS = {
    "classification": 0.15,
    "retrieval": 0.20,
    "tools": 0.20,
    "evidence": 0.15,
    "validation": 0.25,
    "customer_context": 0.05,
}


def evaluate_confidence(state: dict, settings: Settings) -> dict:
    """Score the run and explain each contributing factor."""
    classification = state.get("classification") or {}
    retrieved = state.get("retrieved") or []
    tool_results = state.get("tool_results") or []
    validation = state.get("validation") or {}
    customer = state.get("customer") or {}

    factors: dict[str, float] = {
        "classification": float(classification.get("confidence") or 0.55),
    }
    if retrieved:
        best = max(float(chunk.get("score") or 0.0) for chunk in retrieved)
        factors["retrieval"] = max(0.0, min(1.0, best))
    else:
        factors["retrieval"] = 0.3 if classification.get("needs_documents") else 0.6

    if tool_results:
        successes = sum(1 for payload in tool_results if payload.get("found") is not False and not payload.get("error"))
        factors["tools"] = successes / len(tool_results)
    else:
        factors["tools"] = 1.0

    evidence_count = len(state.get("evidence") or [])
    factors["evidence"] = min(1.0, evidence_count / 3)
    factors["validation"] = 1.0 if validation.get("passed") else 0.0
    if customer:
        factors["customer_context"] = 1.0
    else:
        factors["customer_context"] = 0.0 if classification.get("needs_customer_context") else 0.7

    score = round(sum(WEIGHTS[name] * value for name, value in factors.items()), 3)
    threshold = settings.confidence_threshold
    level = "high" if score >= 0.8 else "medium" if score >= threshold else "low"

    reasons: list[str] = [
        f"Classification confidence {factors['classification']:.2f}.",
        f"Best retrieval score {factors['retrieval']:.2f}.",
        f"{sum(1 for payload in tool_results if payload.get('found') is not False and not payload.get('error'))}"
        f"/{len(tool_results)} tool calls succeeded.",
        f"{evidence_count} evidence item(s) collected.",
    ]
    if not validation.get("passed"):
        reasons.append("Response validation failed: " + "; ".join(validation.get("issues") or ["unknown reason"]))
    if classification.get("needs_customer_context") and not customer:
        reasons.append("Customer context was required but could not be retrieved.")

    return {
        "score": score,
        "level": level,
        "threshold": threshold,
        "factors": {name: round(value, 3) for name, value in factors.items()},
        "reasons": reasons,
    }


def decide_action(state: dict, settings: Settings) -> dict:
    """Return the resolution/escalation decision with a concrete reason."""
    confidence = state.get("confidence") or evaluate_confidence(state, settings)
    validation = state.get("validation") or {}
    classification = state.get("classification") or {}
    refund_payload = next(
        (payload for payload in state.get("tool_results", []) if payload.get("tool") == "check_refund_eligibility"),
        None,
    )

    if not (state.get("draft") or "").strip():
        return {"action": "escalate", "reason": "No response draft was produced."}
    if not validation.get("passed"):
        return {
            "action": "escalate",
            "reason": "Draft failed validation: " + "; ".join(validation.get("issues") or ["unknown issue"]),
        }
    if classification.get("priority") == "Critical":
        return {"action": "escalate", "reason": "Critical priority tickets always require human review."}
    if classification.get("intent") == "account_locked":
        return {
            "action": "escalate",
            "reason": "Unlocking an account requires identity verification by a human agent.",
        }
    ticket_words = (state.get("ticket", {}).get("description") or "").split()
    if len(ticket_words) < 10 and not state.get("customer") and not classification.get("entities", {}).get("order_id"):
        return {
            "action": "escalate",
            "reason": "The ticket does not contain enough detail to identify the customer or the problem.",
        }
    if classification.get("needs_customer_context") and not state.get("customer"):
        return {
            "action": "escalate",
            "reason": "Customer context is required but the customer could not be identified.",
        }
    if refund_payload and not refund_payload.get("eligible"):
        return {
            "action": "escalate",
            "reason": "Refund request needs human judgment: " + "; ".join(refund_payload.get("reasons") or []),
        }
    if float(confidence.get("score") or 0.0) < settings.confidence_threshold:
        return {
            "action": "escalate",
            "reason": f"Confidence {confidence.get('score')} is below the {settings.confidence_threshold} threshold.",
        }
    return {"action": "resolve", "reason": "Evidence is sufficient and the draft passed validation."}
