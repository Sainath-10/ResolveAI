"""Ticket classification.

The classifier returns a category, intent, priority, extracted entities and
routing hints. With an Anthropic key it uses a structured-output call; without
one it falls back to transparent keyword rules so the whole workflow still
runs in offline mode.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel

from config import Settings

logger = logging.getLogger("resolveai.classifier")

CATEGORIES = ("billing", "orders_delivery", "refunds_returns", "technical", "account", "subscription", "general")
CATEGORY_LABELS = {
    "billing": "Billing",
    "orders_delivery": "Orders and delivery",
    "refunds_returns": "Refunds and returns",
    "technical": "Technical issues",
    "account": "Account issues",
    "subscription": "Subscription issues",
    "general": "General enquiries",
}

_CATEGORY_KEYWORDS = {
    "billing": ["charged twice", "duplicate charge", "double charge", "charge", "charged", "payment", "card", "invoice", "billing", "overcharged"],
    "refunds_returns": ["refund", "return", "money back", "send it back", "exchange", "reimburs", "damaged", "arrived broken", "broken"],
    "orders_delivery": ["delivery", "deliver", "shipping", "tracking", "parcel", "package", "carrier", "where is my order", "late", "delay", "arrive"],
    "technical": ["error", "bug", "crash", "not working", "doesn't work", "does not work", "broken page", "api", "429", "loading", "glitch"],
    "account": ["sign in", "sign-in", "log in", "login", "password", "locked", "account", "2fa", "verification", "verify"],
    "subscription": ["subscription", "plan", "renew", "past due", "cancel", "billing cycle", "annual", "monthly"],
}

_PRIORITY_KEYWORDS = {
    "Critical": ["fraud", "unauthori", "stolen", "security breach", "data loss", "legal", "chargeback", "safety"],
    "High": ["charged twice", "duplicate charge", "double charge", "locked", "cannot access", "can't access", "urgent", "damaged", "broken", "twice"],
    "Medium": ["refund", "return", "delay", "late", "past due", "wrong item", "missing", "failed"],
}

_ENTITY_PATTERNS = {
    "customer_id": re.compile(r"\bCUST-(\d{4})\b", re.IGNORECASE),
    "order_id": re.compile(r"\bORD-(\d{4})\b", re.IGNORECASE),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
}


class LLMClassification(BaseModel):
    """Structured output requested from the model."""

    category: Literal["billing", "orders_delivery", "refunds_returns", "technical", "account", "subscription", "general"]
    intent: str
    priority: Literal["Low", "Medium", "High", "Critical"]
    summary: str
    needs_customer_context: bool


def extract_entities(text: str, ticket: dict | None = None) -> dict[str, str]:
    """Pull customer/order IDs and email addresses out of the ticket."""
    ticket = ticket or {}
    entities: dict[str, str] = {}
    for name, pattern in _ENTITY_PATTERNS.items():
        match = pattern.search(text or "")
        if match:
            entities[name] = match.group(0).upper() if name != "email" else match.group(0).lower()
    for field in ("customer_id", "order_id"):
        if ticket.get(field):
            entities[field] = str(ticket[field]).upper()
    return entities


def _score_categories(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {category: sum(2 if keyword in lowered else 0 for keyword in words) for category, words in _CATEGORY_KEYWORDS.items()}


def _intent_for(category: str, text: str) -> str:
    lowered = text.lower()
    if "duplicate" in lowered or "twice" in lowered or "double" in lowered:
        return "duplicate_charge"
    if "damaged" in lowered or "broken" in lowered:
        return "damaged_item"
    if category == "refunds_returns":
        return "refund_request"
    if category == "orders_delivery":
        return "delivery_delay" if any(word in lowered for word in ("late", "delay", "tracking")) else "order_status"
    if category == "account":
        return "account_locked" if "locked" in lowered else "account_access"
    if category == "subscription":
        return "subscription_payment_failure" if any(word in lowered for word in ("past due", "failed")) else "subscription_question"
    if category == "technical":
        return "technical_issue"
    if category == "billing":
        return "billing_question"
    return "general_enquiry"


def _priority_for(text: str, category: str) -> str:
    lowered = text.lower()
    for priority in ("Critical", "High", "Medium"):
        if any(keyword in lowered for keyword in _PRIORITY_KEYWORDS[priority]):
            return priority
    return "Medium" if category == "account" else "Low"


def heuristic_classification(ticket: dict, entities: dict[str, str]) -> dict:
    """Deterministic rules used offline and as the LLM fallback."""
    text = ticket.get("description", "") or ""
    scores = _score_categories(text)
    lowered_text = text.lower()
    if any(word in lowered_text for word in ("refund", "return", "money back", "reimburs")):
        scores["refunds_returns"] += 4
    if any(word in lowered_text for word in ("charged twice", "duplicate charge", "double charge")):
        scores["billing"] += 4
    best_score = max(scores.values(), default=0)
    category = next((name for name in CATEGORIES if scores[name] == best_score and best_score > 0), "general")
    if best_score == 0:
        category = "general"

    words = [word for word in re.findall(r"[a-z]+", lowered_text) if len(word) > 2]
    if len(words) <= 6 and not entities:
        category = "general"
    needs_documents = category in {"billing", "orders_delivery", "refunds_returns", "technical", "account", "subscription"}
    return {
        "category": category,
        "category_label": CATEGORY_LABELS[category],
        "intent": _intent_for(category, text),
        "priority": _priority_for(text, category),
        "entities": entities,
        "needs_customer_context": bool(entities.get("customer_id"))
        or category in {"account", "subscription", "billing", "refunds_returns"},
        "needs_documents": needs_documents,
        "needs_tools": category != "general" or bool(entities.get("order_id")),
        "summary": _summary(text, category),
        "confidence": 0.78 if best_score >= 2 else 0.5,
        "mode": "offline",
    }


def _summary(text: str, category: str) -> str:
    snippet = " ".join((text or "").split())
    if len(snippet) > 140:
        snippet = snippet[:137] + "..."
    return f"{CATEGORY_LABELS[category]} ticket: {snippet or 'no description provided'}"


def _llm_classification(ticket: dict, entities: dict[str, str], settings: Settings) -> dict:
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(model=settings.llm_model, temperature=0, max_tokens=350)
    result = model.with_structured_output(LLMClassification).invoke(
        [
            (
                "system",
                "You classify customer-support tickets for an online store. Pick the single best category: "
                "billing, orders_delivery, refunds_returns, technical, account, subscription, general. "
                "Priority reflects urgency and risk: Critical for fraud or security, High for duplicate charges, "
                "locked accounts or damaged goods, Medium for ordinary refunds and delays, Low otherwise.",
            ),
            (
                "human",
                f"Ticket: {ticket.get('description', '')}\n"
                f"Known customer: {entities.get('customer_id', 'unknown')}\n"
                f"Known order: {entities.get('order_id', 'unknown')}",
            ),
        ]
    )
    category = result.category
    return {
        "category": category,
        "category_label": CATEGORY_LABELS[category],
        "intent": _intent_for(category, ticket.get("description", "")),
        "priority": result.priority,
        "entities": entities,
        "needs_customer_context": result.needs_customer_context or bool(entities.get("customer_id")),
        "needs_documents": category != "general",
        "needs_tools": category != "general" or bool(entities.get("order_id")),
        "summary": result.summary,
        "confidence": 0.88,
        "mode": "llm",
    }


def classify(ticket: dict, settings: Settings) -> dict:
    """Classify one ticket, preferring the model when a key is configured."""
    text = ticket.get("description", "") or ""
    entities = extract_entities(text, ticket)

    if settings.llm_available:
        try:
            return _llm_classification(ticket, entities, settings)
        except Exception as error:  # network/auth failures must not stop resolution
            logger.warning("LLM classification failed, using rules: %s", error)

    return heuristic_classification(ticket, entities)
