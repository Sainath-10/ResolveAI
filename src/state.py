"""Shared LangGraph state types for ResolveAI.

The graph passes one plain dictionary through every node. The TypedDicts below
document what each stage adds; they are deliberately flat and JSON-friendly so
the Streamlit UI can render the state directly.
"""

from __future__ import annotations

from typing import Any, TypedDict


class Ticket(TypedDict, total=False):
    ticket_id: str
    customer_id: str
    order_id: str
    description: str
    channel: str
    created_at: str


class Classification(TypedDict, total=False):
    category: str
    category_label: str
    intent: str
    priority: str
    entities: dict[str, str]
    needs_customer_context: bool
    needs_documents: bool
    needs_tools: bool
    summary: str
    confidence: float
    mode: str


class Evidence(TypedDict, total=False):
    label: str
    kind: str
    detail: str


class Validation(TypedDict, total=False):
    passed: bool
    issues: list[str]
    checks: list[dict[str, Any]]


class Confidence(TypedDict, total=False):
    score: float
    level: str
    threshold: float
    factors: dict[str, float]
    reasons: list[str]


class SupportState(TypedDict, total=False):
    ticket: Ticket
    classification: Classification
    customer: dict[str, Any]
    retrieved: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    evidence: list[Evidence]
    draft: str
    validation: Validation
    confidence: Confidence
    escalation: dict[str, Any]
    route: str
    revision_attempts: int
    react_rounds: int
    final_status: str
    mode: str
    workflow: list[dict[str, Any]]


def new_state(ticket: dict[str, Any]) -> SupportState:
    """Fresh state for one ticket run."""
    return {
        "ticket": dict(ticket),
        "classification": {},
        "customer": {},
        "retrieved": [],
        "tool_results": [],
        "evidence": [],
        "draft": "",
        "validation": {},
        "confidence": {},
        "escalation": {},
        "route": "",
        "revision_attempts": 0,
        "react_rounds": 0,
        "final_status": "",
        "mode": "",
        "workflow": [],
    }
