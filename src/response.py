"""Evidence aggregation, response drafting and validation.

The draft is written only from retrieved policy chunks and tool observations.
Validation then checks that the reply addresses the ticket, cites its evidence,
does not claim an action was completed and does not contradict a tool result
(for example promising a refund the policy check rejected).
"""

from __future__ import annotations

import logging
import re

from config import Settings

logger = logging.getLogger("resolveai.response")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "for", "with", "this", "that", "these", "those",
    "i", "my", "me", "we", "our", "you", "your", "please", "hi", "hello", "thanks", "thank",
    "is", "it", "was", "were", "has", "have", "had", "do", "does", "did", "can", "could",
    "to", "of", "on", "in", "at", "from", "about", "would", "like", "need", "get", "been",
}

_ACTION_CLAIM = re.compile(
    r"\b(?:we|i)\s+(?:have\s+|'ve\s+)?(?:refunded|issued|processed|sent|cancelled|canceled|unlocked|charged)\b"
    r"|\byour\s+(?:refund|account|subscription)\s+(?:has been|was|is now)\s+"
    r"(?:processed|approved|issued|cancelled|canceled|unlocked)\b",
    re.IGNORECASE,
)


def aggregate_evidence(state: dict) -> list[dict]:
    """Merge retrieval chunks and tool sources into a deduplicated list."""
    evidence: list[dict] = []
    seen: set[str] = set()

    for chunk in state.get("retrieved", []):
        label = chunk.get("label") or chunk.get("source", "document")
        if label in seen:
            continue
        seen.add(label)
        evidence.append({"label": label, "kind": "document", "detail": chunk.get("text", "")[:400]})

    for payload in state.get("tool_results", []):
        for source in payload.get("sources") or []:
            label = source.get("label") or source.get("reference") or "tool result"
            if label in seen:
                continue
            seen.add(label)
            evidence.append({"label": label, "kind": source.get("kind", "tool"), "detail": payload.get("tool", "")})

    return evidence


def evidence_text(evidence: list[dict]) -> str:
    return "\n".join(f"[S{index}] {item['label']}: {item['detail']}" for index, item in enumerate(evidence, start=1))


def format_sources(evidence: list[dict]) -> str:
    if not evidence:
        return ""
    lines = [f"- [S{index}] {item['label']}" for index, item in enumerate(evidence, start=1)]
    return "\n\n**Sources**\n" + "\n".join(lines)


def _top_sentences(text: str, limit: int = 2) -> str:
    clean = " ".join((text or "").split())
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    return " ".join(sentences[:limit])


def _order_findings(payload: dict) -> list[str]:
    order = payload.get("order")
    if not isinstance(order, dict):
        return []
    findings = [
        f"- Order {order.get('order_id')}: {order.get('item')} ({order.get('currency')} {order.get('amount')}), "
        f"status '{order.get('status')}'"
        + (f", expected {order.get('eta')}" if order.get("eta") else "")
        + "."
    ]
    if order.get("tracking_number"):
        findings.append(f"  Tracking: {order['carrier']} {order['tracking_number']}.")
    if order.get("delayed"):
        findings.append("  The order is past its estimated delivery date.")
    return findings


def _refund_findings(payload: dict) -> list[str]:
    if "eligible" not in payload:
        return []
    verdict = "eligible" if payload["eligible"] else "not eligible"
    findings = [f"- Refund check for {payload.get('order_id')}: {verdict}."]
    findings.extend(f"  - {reason}" for reason in payload.get("reasons", []))
    if payload.get("eligible"):
        findings.append("  A human agent still has to confirm and process the refund.")
    return findings


def _subscription_findings(payload: dict) -> list[str]:
    subscription = payload.get("subscription")
    if not isinstance(subscription, dict):
        return []
    renewal = subscription.get("renews_on") or "no renewal date"
    return [
        f"- Subscription: {subscription.get('plan')} ({subscription.get('status')}), "
        f"{subscription.get('currency')} {subscription.get('amount')} {subscription.get('billing_cycle')}, "
        f"renews on {renewal}."
    ]


def _status_findings(payload: dict) -> list[str]:
    degraded = payload.get("degraded_services") or []
    incidents = payload.get("incidents") or []
    if not degraded and not incidents:
        return ["- Platform status: all services operational with no recent incidents."]
    parts = []
    if degraded:
        parts.append(f"{', '.join(degraded)} degraded")
    if incidents:
        latest = incidents[0]
        parts.append(f"recent incident {latest.get('incident_id')} ({latest.get('service')}): {latest.get('summary')}")
    return [f"- Platform status: {'; '.join(parts)}."]


def _account_findings(payload: dict) -> list[str]:
    customer = payload.get("customer")
    if isinstance(customer, dict) and customer.get("account_status") not in (None, "active"):
        return [f"- Account status: {customer.get('account_status')} for {customer.get('email')}."]
    return []


def _offline_draft(state: dict, evidence: list[dict]) -> str:
    """Template reply assembled only from observations and retrieved policy."""
    ticket = state.get("ticket", {})
    classification = state.get("classification", {})
    customer = state.get("customer", {})
    name = customer.get("name") or "there"
    description = " ".join((ticket.get("description") or "your request").split())
    if len(description) > 180:
        description = description[:177] + "..."

    findings: list[str] = []
    for payload in state.get("tool_results", []):
        tool = payload.get("tool")
        if tool == "get_order_status":
            findings.extend(_order_findings(payload))
        elif tool == "check_refund_eligibility":
            findings.extend(_refund_findings(payload))
        elif tool == "get_subscription_details":
            findings.extend(_subscription_findings(payload))
        elif tool == "get_recent_system_status":
            findings.extend(_status_findings(payload))
        elif tool == "get_customer_profile":
            findings.extend(_account_findings(payload))

    guidance = []
    for index, item in enumerate(evidence, start=1):
        if item.get("kind") == "document" and len(guidance) < 2:
            guidance.append(f"- {_top_sentences(item.get('detail', ''), 2)} [S{index}]")

    intent = classification.get("intent", "general_enquiry")
    next_steps = {
        "duplicate_charge": "Duplicate card authorizations usually clear within a few business days. If both charges have settled, our billing team will review the duplicate and confirm the next step.",
        "refund_request": "A support agent will review this against the refund policy and contact you with the outcome.",
        "damaged_item": "Please keep the damaged item and its packaging. A support agent will confirm a replacement or refund after reviewing the claim.",
        "delivery_delay": "We will check with the carrier and follow up with an updated delivery estimate.",
        "account_locked": "Unlocking an account requires identity verification, so a support specialist will help you directly.",
        "subscription_payment_failure": "You can update the payment method in your billing settings. The plan keeps a 7-day grace period before paid features are suspended.",
        "subscription_question": "A support agent will confirm the subscription details and any changes to your plan.",
        "technical_issue": "Please try the troubleshooting steps above. If the problem continues, a support specialist will investigate further.",
    }.get(intent, "A support specialist will review your message and follow up.")

    paragraphs = [
        f"Hi {name},",
        f"Thanks for getting in touch about: {description}",
    ]
    if findings:
        paragraphs.append("Here is what I found:\n" + "\n".join(findings))
    if guidance:
        paragraphs.append("What the policy says:\n" + "\n".join(guidance))
    paragraphs.append(f"Next step: {next_steps}")
    paragraphs.append("I have not made any changes to your order, account or subscription.")
    body = "\n\n".join(paragraphs)
    return body + format_sources(evidence)


def _llm_draft(state: dict, settings: Settings, evidence: list[dict], critique: list[str] | None) -> str:
    from langchain_anthropic import ChatAnthropic

    ticket = state.get("ticket", {})
    classification = state.get("classification", {})
    customer = state.get("customer", {})
    model = ChatAnthropic(model=settings.llm_model, temperature=0.2, max_tokens=700)
    prompt = (
        f"Customer ticket: {ticket.get('description', '')}\n"
        f"Category: {classification.get('category_label', 'unknown')}; priority: {classification.get('priority', 'unknown')}\n"
        f"Customer: {customer.get('name', 'unknown')} ({customer.get('customer_id', 'unknown')}), "
        f"account {customer.get('account_status', 'unknown')}\n\n"
        f"EVIDENCE (cite as [S1], [S2]...):\n{evidence_text(evidence)}\n\n"
        f"Validation issues to fix: {critique or 'none'}"
    )
    response = model.invoke(
        [
            (
                "system",
                "You are a customer-support agent for an online store. Write a concise, professional reply "
                "to the ticket using ONLY the evidence provided. Cite the supporting evidence inline as [S#]. "
                "Do not invent order, customer or policy facts. Never claim that a refund, cancellation, "
                "unlock or message was completed; say a human will confirm it. If the evidence is missing, "
                "say what could not be verified and ask for the specific detail you need. "
                "End with a clear next step. Do not add a sources list; the application appends it.",
            ),
            ("human", prompt),
        ]
    )
    content = response.content
    text = content if isinstance(content, str) else "".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )
    return text.strip() + format_sources(evidence)


def draft_response(state: dict, settings: Settings, critique: list[str] | None = None) -> str:
    """Draft the customer reply, falling back to the template offline."""
    evidence = state.get("evidence") or aggregate_evidence(state)
    if settings.llm_available:
        try:
            return _llm_draft(state, settings, evidence, critique)
        except Exception as error:
            logger.warning("LLM draft failed, using template: %s", error)
    draft = _offline_draft(state, evidence)
    if critique:
        draft += "\n\n_This draft was revised after validation: " + "; ".join(critique) + "_"
    return draft


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def validate_response(draft: str, state: dict, settings: Settings) -> dict:
    """Check the draft for completeness, grounding and unsafe claims."""
    evidence = state.get("evidence") or aggregate_evidence(state)
    ticket_tokens = _meaningful_tokens(state.get("ticket", {}).get("description", ""))
    draft_tokens = _meaningful_tokens(draft)
    checks: list[dict] = []
    issues: list[str] = []

    def check(name: str, passed: bool, message: str) -> None:
        checks.append({"name": name, "passed": passed})
        if not passed:
            issues.append(message)

    check("response_present", len(draft.strip()) >= 80, "The draft is missing or too short.")
    overlap = len(ticket_tokens & draft_tokens)
    check("addresses_issue", overlap >= 2, "The draft does not clearly address the ticket's wording.")
    check("cites_evidence", not evidence or "[S" in draft, "The draft does not cite any retrieved evidence.")
    check("no_completed_actions", not _ACTION_CLAIM.search(draft), "The draft claims an action has already been completed.")

    refund_payload = next(
        (payload for payload in state.get("tool_results", []) if payload.get("tool") == "check_refund_eligibility"),
        None,
    )
    if refund_payload and not refund_payload.get("eligible"):
        contradiction = re.search(r"\b(eligible|qualify)\b[^.]{0,40}\brefund\b", draft, re.IGNORECASE)
        check(
            "policy_consistent",
            contradiction is None,
            "The draft says the customer is eligible although the refund check rejected the order.",
        )
    else:
        check("policy_consistent", True, "")

    entity_order = state.get("classification", {}).get("entities", {}).get("order_id")
    order_payloads = [payload for payload in state.get("tool_results", []) if payload.get("tool") == "get_order_status"]
    check(
        "order_context_present",
        not entity_order or any(payload.get("found") for payload in order_payloads),
        "An order ID was mentioned but its details could not be retrieved.",
    )

    return {"passed": not issues, "issues": issues, "checks": checks}
