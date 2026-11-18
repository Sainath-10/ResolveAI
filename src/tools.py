"""The eight functional customer-support tools.

Each tool reads the demonstration datasets (or the Chroma knowledge base) and
returns JSON, including a ``sources`` list that becomes the evidence shown in
the UI. Errors are returned as structured values so the agent can observe a
failure instead of crashing.

No tool sends a message, issues a refund or changes a customer account.
``create_support_escalation`` is the only tool that writes, and it only adds a
human-review record.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

from config import get_settings
from retriever import search

logger = logging.getLogger("resolveai.tools")


def _data_dir() -> Path:
    return get_settings().data_dir


@lru_cache(maxsize=8)
def _load(name: str) -> list | dict:
    return json.loads((_data_dir() / name).read_text(encoding="utf-8"))


def _records(name: str) -> list[dict]:
    payload = _load(name)
    return payload if isinstance(payload, list) else []


def _find(records: list[dict], key: str, value: str) -> dict | None:
    wanted = (value or "").strip().upper()
    for record in records:
        if str(record.get(key, "")).upper() == wanted:
            return record
    return None


def _reference_now() -> datetime:
    """Demo clock: the status snapshot date, so eligibility is deterministic."""
    status = _load("system_status.json")
    updated = status.get("updated_at") if isinstance(status, dict) else None
    if updated:
        return datetime.fromisoformat(updated.replace("Z", "+00:00"))
    return datetime.now(timezone.utc)


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _source(kind: str, label: str, reference: str) -> dict:
    return {"kind": kind, "label": label, "reference": reference}


def _missing(tool_name: str, message: str) -> str:
    return _dump({"tool": tool_name, "found": False, "error": message, "sources": []})


@tool
def search_knowledge_base(query: str, category: str = "", limit: int = 3) -> str:
    """Search the support knowledge base for policies, FAQs and troubleshooting steps.

    Use this for refund rules, delivery expectations, subscription terms or
    technical guidance. Optionally pass a category such as refunds_returns,
    orders_delivery, subscription or technical, and a limit between 1 and 5.
    """
    limit = max(1, min(int(limit or 3), 5))
    hits = search(query, k=limit, category=category or None)
    results = [
        {
            "id": hit["id"],
            "source": hit["source"],
            "section": hit["section"],
            "page": hit["page"],
            "text": hit["text"],
            "score": hit["score"],
        }
        for hit in hits
    ]
    sources = [_source("document", hit["label"], hit["source"]) for hit in hits]
    return _dump({"tool": "search_knowledge_base", "found": bool(results), "results": results, "sources": sources})


@tool
def get_recent_system_status() -> str:
    """Get current service availability and recent incidents.

    Use this for technical problems, login failures or delivery-tracking issues
    that might be caused by a platform incident rather than the customer's account.
    """
    status = _load("system_status.json")
    if not isinstance(status, dict):
        return _missing("get_recent_system_status", "System status data is unavailable.")
    incidents = status.get("incidents", [])
    degraded = [service["name"] for service in status.get("services", []) if service.get("status") != "operational"]
    return _dump(
        {
            "tool": "get_recent_system_status",
            "found": True,
            "updated_at": status.get("updated_at"),
            "services": status.get("services", []),
            "degraded_services": degraded,
            "incidents": incidents,
            "sources": [_source("status", "Service status snapshot", "system_status.json")],
        }
    )


@tool
def get_customer_recent_activity(user_id: str) -> str:
    """Look up a customer's recent logins, orders, billing events and support contacts.

    Use this to check whether the ticket is consistent with what is on record.
    Requires the customer ID, for example CUST-1001.
    """
    customer = _find(_records("customers.json"), "customer_id", user_id)
    if customer is None:
        return _missing("get_customer_recent_activity", f"No customer found with ID {user_id or '(missing)'}.")
    activity = customer.get("recent_activity", [])
    return _dump(
        {
            "tool": "get_customer_recent_activity",
            "found": True,
            "customer_id": customer["customer_id"],
            "name": customer.get("name"),
            "recent_activity": activity,
            "sources": [_source("customer", f"Customer activity ({customer['customer_id']})", "customers.json")],
        }
    )


@tool
def get_customer_profile(user_id: str) -> str:
    """Retrieve a customer's name, email, tier and account status.

    Use this before discussing account access, subscriptions or eligibility.
    Requires the customer ID, for example CUST-1001.
    """
    customer = _find(_records("customers.json"), "customer_id", user_id)
    if customer is None:
        return _missing("get_customer_profile", f"No customer found with ID {user_id or '(missing)'}.")
    profile = {
        "customer_id": customer["customer_id"],
        "name": customer.get("name"),
        "email": customer.get("email"),
        "tier": customer.get("tier"),
        "account_status": customer.get("account_status"),
        "joined_at": customer.get("joined_at"),
    }
    return _dump(
        {
            "tool": "get_customer_profile",
            "found": True,
            "customer": profile,
            "sources": [_source("customer", f"Customer profile ({customer['customer_id']})", "customers.json")],
        }
    )


@tool
def get_order_status(order_id: str) -> str:
    """Retrieve an order's item, payment amount, delivery status, ETA and tracking.

    Use this whenever the customer mentions an order ID, a delivery problem or
    a charge that needs to be matched to a purchase.
    """
    order = _find(_records("orders.json"), "order_id", order_id)
    if order is None:
        return _missing("get_order_status", f"No order found with ID {order_id or '(missing)'}.")
    delayed = False
    eta = order.get("eta")
    if eta and order.get("status") in {"in_transit", "processing"}:
        try:
            delayed = datetime.fromisoformat(str(eta)).date() < _reference_now().date()
        except ValueError:
            delayed = False
    summary = {
        "order_id": order["order_id"],
        "customer_id": order.get("customer_id"),
        "item": order.get("item"),
        "amount": order.get("amount"),
        "currency": order.get("currency"),
        "status": order.get("status"),
        "placed_at": order.get("placed_at"),
        "delivered_at": order.get("delivered_at"),
        "eta": eta,
        "carrier": order.get("carrier"),
        "tracking_number": order.get("tracking_number"),
        "refund_status": order.get("refund_status"),
        "delayed": delayed,
    }
    return _dump(
        {
            "tool": "get_order_status",
            "found": True,
            "order": summary,
            "note": "The order is past its estimated delivery date." if delayed else "",
            "sources": [_source("order", f"Order {order['order_id']} ({order.get('item')})", "orders.json")],
        }
    )


@tool
def check_refund_eligibility(order_id: str) -> str:
    """Check whether an order is eligible for a refund under the refund policy.

    Use this before discussing refunds. This tool only evaluates the policy; it
    never issues a refund or promises that one has been approved.
    """
    order = _find(_records("orders.json"), "order_id", order_id)
    if order is None:
        return _missing("check_refund_eligibility", f"No order found with ID {order_id or '(missing)'}.")

    reasons: list[str] = []
    delivered_at = order.get("delivered_at")
    days_since: int | None = None
    if order.get("status") != "delivered" or not delivered_at:
        reasons.append(f"Order status is '{order.get('status')}', so it has not been delivered yet.")
    else:
        delivered = datetime.fromisoformat(str(delivered_at).replace("Z", "+00:00"))
        days_since = (_reference_now().date() - delivered.date()).days
        if days_since > 30:
            reasons.append(f"Delivered {days_since} days ago, outside the 30-day refund window.")
    if order.get("digital"):
        reasons.append("Digital products are not refundable.")
    if order.get("final_sale"):
        reasons.append("The item is marked final sale.")
    if order.get("refund_status") in {"requested", "refunded"}:
        reasons.append(f"A refund or claim is already '{order.get('refund_status')}' for this order.")

    eligible = not reasons
    return _dump(
        {
            "tool": "check_refund_eligibility",
            "found": True,
            "order_id": order["order_id"],
            "eligible": eligible,
            "reasons": reasons,
            "days_since_delivery": days_since,
            "policy": {"source": "refund_policy.md", "section": "Eligibility"},
            "next_step": "A human agent must confirm and process any refund."
            if eligible
            else "Explain the policy reason; a human reviews exceptions.",
            "sources": [_source("policy", "Refund policy - Eligibility", "refund_policy.md")],
        }
    )


@tool
def get_subscription_details(user_id: str) -> str:
    """Retrieve a customer's subscription plan, status, amount and renewal date.

    Use this for cancellation, renewal, plan-change or failed-payment questions.
    """
    subscription = _find(_records("subscriptions.json"), "customer_id", user_id)
    if subscription is None:
        return _missing("get_subscription_details", f"No subscription found for customer {user_id or '(missing)'}.")
    return _dump(
        {
            "tool": "get_subscription_details",
            "found": True,
            "subscription": subscription,
            "sources": [_source("subscription", f"Subscription ({subscription.get('plan')})", "subscriptions.json")],
        }
    )


@tool
def create_support_escalation(ticket_id: str, reason: str) -> str:
    """Create a persistent human-review record for a ticket.

    Use this only when the workflow cannot safely resolve the ticket. Existing
    open escalations for the same ticket are reused instead of duplicated.
    """
    ticket = (ticket_id or "").strip().upper()
    reason = " ".join((reason or "").split())
    if not ticket:
        return _missing("create_support_escalation", "A ticket ID is required.")
    if not reason:
        return _missing("create_support_escalation", "An escalation reason is required.")

    path = _data_dir() / "escalations.json"
    try:
        records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except json.JSONDecodeError:
        records = []

    for record in records:
        if record.get("ticket_id") == ticket and record.get("status") == "open":
            return _dump(
                {
                    "tool": "create_support_escalation",
                    "found": True,
                    "created": False,
                    "escalation": record,
                    "note": "An open escalation already exists for this ticket.",
                    "sources": [],
                }
            )

    escalation = {
        "escalation_id": f"ESC-{len(records) + 1001}",
        "ticket_id": ticket,
        "reason": reason,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    records.append(escalation)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return _dump(
        {
            "tool": "create_support_escalation",
            "found": True,
            "created": True,
            "escalation": escalation,
            "sources": [_source("escalation", f"Escalation {escalation['escalation_id']}", "escalations.json")],
        }
    )


SUPPORT_TOOLS = [
    search_knowledge_base,
    get_recent_system_status,
    get_customer_recent_activity,
    get_customer_profile,
    get_order_status,
    check_refund_eligibility,
    get_subscription_details,
    create_support_escalation,
]
TOOL_BY_NAME = {academic_tool.name: academic_tool for academic_tool in SUPPORT_TOOLS}
