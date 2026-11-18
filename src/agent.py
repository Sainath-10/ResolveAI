"""LangGraph orchestration and the ReAct execution loop.

    START -> classify -> gather_context -> route
        route -> retrieve_knowledge -> (react_agent) -> draft
        route -> react_agent -> draft
        draft -> validate -> confidence -> resolve | escalate -> END

The graph performs real tool calls: with an Anthropic key the model chooses
tools through ``bind_tools`` and observes their JSON results; without a key a
deterministic planner selects the same real tools from the classification and
entities, and the reply is clearly labelled offline.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from config import Settings, get_settings
from classifier import classify
from escalation import decide_action, evaluate_confidence
from knowledge_base import ensure_index
from response import aggregate_evidence, draft_response, validate_response
from retriever import search
from state import SupportState, new_state
from tools import SUPPORT_TOOLS, TOOL_BY_NAME

logger = logging.getLogger("resolveai.agent")


def _log(state: dict, stage: str, detail: str) -> list[dict]:
    return [*state.get("workflow", []), {"stage": stage, "detail": detail}]


def create_chat_model(tools: list | None = None, settings: Settings | None = None):
    """Return a configured Claude model, or ``None`` in offline mode."""
    settings = settings or get_settings()
    if not settings.llm_available:
        return None
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(model=settings.llm_model, temperature=0, max_tokens=700)
    return model.bind_tools(tools) if tools else model


def _execute_tool(name: str, args: dict) -> str:
    """Run one registered tool and always return a JSON string."""
    academic_tool = TOOL_BY_NAME.get(name)
    if academic_tool is None:
        return json.dumps({"tool": name, "found": False, "error": f"Unknown tool: {name}", "sources": []})
    try:
        result = academic_tool.invoke(args or {})
    except Exception as error:  # a failing tool is an observation, not a crash
        logger.exception("Tool %s failed", name)
        return json.dumps({"tool": name, "found": False, "error": str(error), "sources": []})
    return result if isinstance(result, str) else json.dumps(result, default=str)


def _parse(payload: str) -> dict:
    try:
        parsed = json.loads(payload)
        return parsed if isinstance(parsed, dict) else {"found": False, "error": "Malformed tool result"}
    except json.JSONDecodeError:
        return {"found": False, "error": "Malformed tool result"}


def classify_ticket(state: dict) -> dict:
    """Stage 2: category, intent, priority, entities and routing hints."""
    classification = classify(state.get("ticket", {}), get_settings())
    detail = (
        f"{classification['category_label']} / {classification['priority']} priority / "
        f"{classification['intent']} ({classification['mode']})"
    )
    return {
        "classification": classification,
        "mode": classification.get("mode", "offline"),
        "workflow": _log(state, "classify", detail),
    }


def gather_context(state: dict) -> dict:
    """Stage 3: pull the customer profile and recent activity when relevant."""
    classification = state.get("classification", {})
    entities = classification.get("entities", {})
    customer_id = entities.get("customer_id", "")
    results = list(state.get("tool_results", []))
    customer: dict[str, Any] = {}

    if classification.get("needs_customer_context") and customer_id:
        for name, args in (
            ("get_customer_profile", {"user_id": customer_id}),
            ("get_customer_recent_activity", {"user_id": customer_id}),
        ):
            payload = _parse(_execute_tool(name, args))
            results.append(payload)
            if name == "get_customer_profile" and payload.get("found"):
                customer = payload.get("customer", {})
        detail = f"loaded profile and activity for {customer_id}"
    elif classification.get("needs_customer_context"):
        detail = "no customer ID supplied; context unavailable"
    else:
        detail = "customer context not required for this category"

    needs_documents = bool(classification.get("needs_documents"))
    needs_tools = bool(classification.get("needs_tools"))
    if needs_documents and needs_tools:
        route = "both"
    elif needs_documents:
        route = "knowledge"
    else:
        route = "react"

    return {
        "customer": customer,
        "tool_results": results,
        "route": route,
        "workflow": _log(state, "gather_context", f"{detail}; {route} path"),
    }


def _route_selector(state: dict) -> str:
    """Pure routing function for the conditional edge after context gathering."""
    return state.get("route", "knowledge")


def retrieve_knowledge(state: dict) -> dict:
    """Stage 5A: semantic policy retrieval from ChromaDB."""
    settings = get_settings()
    ticket = state.get("ticket", {})
    category = state.get("classification", {}).get("category")
    query = ticket.get("description", "")
    chunks = search(query, k=settings.top_k, category=category, settings=settings)
    retrieved = [
        {
            "id": chunk["id"],
            "text": chunk["text"],
            "source": chunk["source"],
            "title": chunk["title"],
            "section": chunk["section"],
            "page": chunk["page"],
            "score": chunk["score"],
            "label": chunk["label"],
        }
        for chunk in chunks
    ]
    evidence = aggregate_evidence({**state, "retrieved": retrieved})
    return {
        "retrieved": retrieved,
        "evidence": evidence,
        "workflow": _log(state, "retrieve", f"{len(retrieved)} policy chunks retrieved"),
    }


def _offline_plan(state: dict) -> list[dict]:
    """Deterministic tool selection for offline mode.

    Tools are chosen from the classification and extracted entities, so
    different tickets receive different investigations. Every call still hits
    a real dataset lookup.
    """
    classification = state.get("classification", {})
    entities = classification.get("entities", {})
    ticket = state.get("ticket", {})
    category = classification.get("category")
    intent = classification.get("intent")
    order_id = entities.get("order_id") or ticket.get("order_id")
    customer_id = entities.get("customer_id") or ticket.get("customer_id")
    description = ticket.get("description", "")

    calls: list[dict] = []

    def add(name: str, args: dict) -> None:
        calls.append({"name": name, "args": args})

    if order_id:
        add("get_order_status", {"order_id": order_id})
        if category == "refunds_returns" or intent in {"refund_request", "damaged_item"}:
            add("check_refund_eligibility", {"order_id": order_id})
    if category == "subscription" and customer_id:
        add("get_subscription_details", {"user_id": customer_id})
    if category in {"account", "technical"}:
        add("get_recent_system_status", {})
    if not state.get("retrieved"):
        add("search_knowledge_base", {"query": description, "limit": 3})
    if not calls:
        add("search_knowledge_base", {"query": description, "limit": 3})
    return calls


def _react_llm(state: dict, settings: Settings) -> tuple[list[dict], int, str]:
    """Genuine ReAct loop: reason, act, observe, repeat (bounded)."""
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    model = create_chat_model(SUPPORT_TOOLS, settings)
    classification = state.get("classification", {})
    customer = state.get("customer", {})
    ticket = state.get("ticket", {})

    system = SystemMessage(
        content=(
            "You are ResolveAI, a customer-support agent. Investigate the ticket by calling the "
            "available tools; choose only the tools whose data you actually need. After each result, "
            "decide whether another tool is required. Never invent order, account or policy facts. "
            "Do not claim an action (refund, cancellation, unlock) was completed.\n"
            f"Customer: {customer.get('customer_id', 'unknown')} ({customer.get('name', 'unknown')}), "
            f"account {customer.get('account_status', 'unknown')}\n"
            f"Order: {classification.get('entities', {}).get('order_id', 'unknown')}\n"
            f"Category: {classification.get('category_label', 'unknown')}"
        )
    )
    human = HumanMessage(content=f"Ticket {ticket.get('ticket_id', '')}: {ticket.get('description', '')}")
    messages = [system, human]
    results: list[dict] = []
    rounds = 0
    final_text = ""

    while rounds < settings.max_react_rounds:
        response = model.invoke(messages)
        messages.append(response)
        content = response.content
        final_text = content if isinstance(content, str) else "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
        calls = getattr(response, "tool_calls", None) or []
        if not calls:
            break
        for call in calls:
            name = call.get("name", "")
            payload_text = _execute_tool(name, call.get("args") or {})
            messages.append(
                ToolMessage(
                    content=payload_text,
                    tool_call_id=call.get("id") or f"call_{rounds}_{len(results)}",
                    name=name,
                )
            )
            results.append(_parse(payload_text))
        rounds += 1

    return results, rounds, final_text


def react_agent(state: dict) -> dict:
    """Stage 5B: model-driven tool selection with an offline planner fallback."""
    settings = get_settings()
    results = list(state.get("tool_results", []))
    started_with = len(results)
    rounds = 0
    mode = "offline"
    note = ""

    if settings.llm_available:
        try:
            new_results, rounds, _ = _react_llm(state, settings)
            results.extend(new_results)
            mode = "llm"
        except Exception as error:  # provider failures must not break resolution
            logger.warning("ReAct model call failed, using offline planner: %s", error)
            note = " (model unavailable, planner fallback)"

    if mode == "offline":
        for call in _offline_plan(state):
            results.append(_parse(_execute_tool(call["name"], call["args"])))
        rounds = 1

    evidence = aggregate_evidence({**state, "tool_results": results})
    detail = f"{len(results) - started_with} tool call(s), {rounds} round(s), {mode}{note}"
    return {
        "tool_results": results,
        "evidence": evidence,
        "react_rounds": rounds,
        "mode": mode,
        "workflow": _log(state, "react", detail),
    }


def draft_reply(state: dict) -> dict:
    """Stage 7: draft a grounded reply from the collected evidence."""
    settings = get_settings()
    prepared = {**state, "evidence": aggregate_evidence(state)}
    draft = draft_response(prepared, settings)
    return {
        "draft": draft,
        "evidence": prepared["evidence"],
        "workflow": _log(state, "draft", f"{len(draft)} characters drafted"),
    }


def validate(state: dict) -> dict:
    """Stage 8: validate the draft, allowing one corrective rewrite."""
    settings = get_settings()
    validation = validate_response(state.get("draft", ""), state, settings)
    attempts = int(state.get("revision_attempts", 0))

    if not validation["passed"] and attempts < 1 and settings.llm_available:
        critique = validation["issues"]
        revised = draft_response(state, settings, critique)
        revalidation = validate_response(revised, state, settings)
        return {
            "draft": revised,
            "validation": revalidation,
            "revision_attempts": attempts + 1,
            "workflow": _log(state, "validate", f"revised once, {len(revalidation['issues'])} issue(s) left"),
        }

    detail = "passed all checks" if validation["passed"] else f"{len(validation['issues'])} issue(s)"
    return {
        "validation": validation,
        "revision_attempts": attempts,
        "workflow": _log(state, "validate", detail),
    }


def assess_confidence(state: dict) -> dict:
    """Stage 9: score confidence from observable factors."""
    confidence = evaluate_confidence(state, get_settings())
    return {
        "confidence": confidence,
        "workflow": _log(state, "confidence", f"{confidence['score']:.2f} ({confidence['level']})"),
    }


def make_decision(state: dict) -> dict:
    """Stage 10 routing: resolve or send the ticket to a human."""
    decision = decide_action(state, get_settings())
    return {
        "decision": decision["action"],
        "final_status": "escalated" if decision["action"] == "escalate" else "resolved",
        "workflow": _log(state, "decide", decision["reason"]),
    }


def escalate_ticket(state: dict) -> dict:
    """Escalation path: persist a human-review record and keep the draft."""
    decision = decide_action(state, get_settings())
    ticket_id = state.get("ticket", {}).get("ticket_id") or "UNKNOWN"
    payload = _parse(
        _execute_tool("create_support_escalation", {"ticket_id": ticket_id, "reason": decision["reason"]})
    )
    results = [*state.get("tool_results", []), payload]
    evidence = aggregate_evidence({**state, "tool_results": results})
    escalation = payload.get("escalation", {})
    return {
        "tool_results": results,
        "evidence": evidence,
        "escalation": escalation,
        "final_status": "escalated",
        "workflow": _log(
            state,
            "escalate",
            f"{escalation.get('escalation_id', 'no record')}: {decision['reason']}",
        ),
    }


def resolve_ticket(state: dict) -> dict:
    """Resolution path: the draft is ready and passed the checks."""
    return {
        "final_status": "resolved",
        "workflow": _log(state, "resolve", "draft response ready for the support team"),
    }


def _after_retrieve(state: dict) -> str:
    return "react" if state.get("route") == "both" else "draft"


def _after_decision(state: dict) -> str:
    return "escalate" if state.get("final_status") == "escalated" else "resolve"


def build_graph():
    """Compile the ResolveAI LangGraph workflow."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(SupportState)
    graph.add_node("classify", classify_ticket)
    graph.add_node("gather_context", gather_context)
    graph.add_node("retrieve", retrieve_knowledge)
    graph.add_node("react", react_agent)
    graph.add_node("draft", draft_reply)
    graph.add_node("validate", validate)
    graph.add_node("confidence", assess_confidence)
    graph.add_node("decide", make_decision)
    graph.add_node("resolve", resolve_ticket)
    graph.add_node("escalate", escalate_ticket)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "gather_context")
    graph.add_conditional_edges(
        "gather_context",
        _route_selector,
        {"knowledge": "retrieve", "react": "react", "both": "retrieve"},
    )
    graph.add_conditional_edges("retrieve", _after_retrieve, {"react": "react", "draft": "draft"})
    graph.add_edge("react", "draft")
    graph.add_edge("draft", "validate")
    graph.add_edge("validate", "confidence")
    graph.add_edge("confidence", "decide")
    graph.add_conditional_edges("decide", _after_decision, {"resolve": "resolve", "escalate": "escalate"})
    graph.add_edge("resolve", END)
    graph.add_edge("escalate", END)
    return graph.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


def process_ticket(ticket: dict, settings: Settings | None = None) -> dict:
    """Run one ticket through the graph and return the final state."""
    settings = settings or get_settings()
    ensure_index(settings)
    state = new_state(ticket)
    config = {"configurable": {"thread_id": ticket.get("ticket_id") or "ticket"}}
    return dict(get_graph().invoke(state, config))
