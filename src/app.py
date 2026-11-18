"""ResolveAI Streamlit interface.

Paste or load a support ticket, run it through the LangGraph workflow and
inspect the real classification, tool results, retrieved policy sources,
draft, validation checks, confidence factors and the final resolution or
escalation status.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent import process_ticket  # noqa: E402
from config import get_settings  # noqa: E402
from knowledge_base import build_index, ensure_index, index_status  # noqa: E402

st.set_page_config(page_title="ResolveAI - Support Resolution", page_icon="🎫", layout="wide")

settings = get_settings()
errors = settings.validation_errors()
if errors:
    st.error("Configuration problem:\n\n" + "\n".join(f"- {error}" for error in errors))
    st.stop()


@st.cache_resource(show_spinner="Preparing the knowledge base...")
def prepare_index() -> dict:
    return ensure_index(settings)


index_info = prepare_index()
samples = json.loads((settings.data_dir / "tickets.json").read_text(encoding="utf-8"))
sample_by_id = {sample["ticket_id"]: sample for sample in samples}

st.title("ResolveAI")
st.caption(
    "Customer-support resolution with LangGraph, RAG, a ReAct tool loop and human-in-the-loop escalation. "
    "All data is fictional demonstration data."
)

with st.sidebar:
    st.subheader("Workflow settings")
    for label, value in settings.describe().items():
        st.caption(f"**{label}:** {value}")
    st.caption(f"**Index:** {index_info['status']} - {index_info['counts']['chunks']} chunks")

    if st.button("Rebuild knowledge index", width="stretch"):
        with st.spinner("Re-embedding policy documents..."):
            rebuilt = build_index(force=True, settings=settings)
        st.cache_resource.clear()
        st.success(f"Index rebuilt: {rebuilt['counts']}")
        st.rerun()

    st.divider()
    choice = st.selectbox("Load a sample ticket", ["-"] + list(sample_by_id), key="sample_choice")
    if choice != "-" and st.button("Fill the form", width="stretch"):
        sample = sample_by_id[choice]
        st.session_state["ticket_id"] = sample["ticket_id"]
        st.session_state["customer_id"] = sample.get("customer_id") or ""
        st.session_state["order_id"] = sample.get("order_id") or ""
        st.session_state["description"] = sample["description"]
        st.rerun()

    with st.expander("Sample tickets"):
        st.dataframe(
            pd.DataFrame(samples)[["ticket_id", "customer_id", "order_id", "description"]],
            hide_index=True,
            width="stretch",
        )

    st.divider()
    st.caption(
        "Northgate Store, its customers, orders, subscriptions and policies are fictional sample data "
        "created for this demo. The advisor drafts replies; it never sends them or performs refunds."
    )

st.subheader("Ticket")
with st.form("ticket_form"):
    col_a, col_b, col_c = st.columns(3)
    ticket_id = col_a.text_input("Ticket ID", key="ticket_id", placeholder="TKT-3001")
    customer_id = col_b.text_input("Customer ID", key="customer_id", placeholder="CUST-1001")
    order_id = col_c.text_input("Order ID (optional)", key="order_id", placeholder="ORD-2041")
    description = st.text_area(
        "Issue description",
        key="description",
        height=140,
        placeholder="Describe the customer's problem in their own words...",
    )
    submitted = st.form_submit_button("Run ResolveAI", type="primary")

if submitted:
    if not description.strip():
        st.warning("Enter a ticket description before running the workflow.")
    else:
        ticket = {
            "ticket_id": ticket_id.strip() or "TKT-LOCAL",
            "customer_id": customer_id.strip(),
            "order_id": order_id.strip(),
            "description": description.strip(),
            "channel": "web",
        }
        with st.spinner("Classifying, retrieving policy and calling tools..."):
            st.session_state["result"] = process_ticket(ticket, settings)

result = st.session_state.get("result")
if not result:
    st.info("Choose a sample ticket in the sidebar or write one above, then run the workflow.")
    st.stop()

classification = result.get("classification", {})
confidence = result.get("confidence", {})
validation = result.get("validation", {})
escalation = result.get("escalation", {})
mode = result.get("mode", "offline")

metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Category", classification.get("category_label", "unknown"))
metric_b.metric("Priority", classification.get("priority", "unknown"))
metric_c.metric("Confidence", f"{float(confidence.get('score') or 0):.2f}")
metric_d.metric("Status", str(result.get("final_status", "")).title())

if result.get("final_status") == "escalated":
    st.error(
        f"Escalated to human review ({escalation.get('escalation_id', 'no record')}): "
        f"{escalation.get('reason', 'reason unavailable')}"
    )
else:
    st.success("Resolved with a validated draft response. Nothing has been sent to the customer.")

with st.expander("Workflow stages", expanded=True):
    for step in result.get("workflow", []):
        st.markdown(f"**{step['stage']}** - {step['detail']}")
    st.caption(f"Execution mode: {mode}")

with st.expander("Classification and customer context"):
    col_left, col_right = st.columns(2)
    with col_left:
        st.write(
            {
                "category": classification.get("category"),
                "intent": classification.get("intent"),
                "priority": classification.get("priority"),
                "entities": classification.get("entities", {}),
                "needs_documents": classification.get("needs_documents"),
                "needs_tools": classification.get("needs_tools"),
            }
        )
        st.caption(classification.get("summary", ""))
    with col_right:
        if result.get("customer"):
            st.write(result["customer"])
        else:
            st.caption("No customer profile was retrieved for this ticket.")

tool_results = result.get("tool_results", [])
with st.expander(f"Tool calls ({len(tool_results)})"):
    if not tool_results:
        st.caption("No tools were called for this ticket.")
    for payload in tool_results:
        tool_name = payload.get("tool", "tool")
        with st.expander(f"{tool_name} - found={payload.get('found')}"):
            st.json(payload)

retrieved = result.get("retrieved", [])
with st.expander(f"Retrieved policy sources ({len(retrieved)})"):
    if not retrieved:
        st.caption("No policy chunks were retrieved.")
    for chunk in retrieved:
        st.markdown(f"**{chunk.get('label')}** - score {chunk.get('score')}")
        st.caption(" ".join(chunk.get("text", "").split())[:400])

evidence = result.get("evidence", [])
with st.expander(f"Evidence used ({len(evidence)})"):
    for position, item in enumerate(evidence, start=1):
        st.markdown(f"**[S{position}]** {item.get('label')} ({item.get('kind')})")

st.subheader("Draft response")
st.markdown(result.get("draft") or "_No draft was generated._")

with st.expander("Validation", expanded=not validation.get("passed")):
    if validation.get("passed"):
        st.success("All validation checks passed.")
    else:
        for issue in validation.get("issues", []):
            st.warning(issue)
    checks = validation.get("checks", [])
    if checks:
        st.dataframe(pd.DataFrame(checks), hide_index=True, width="stretch")

with st.expander("Confidence assessment"):
    score = float(confidence.get("score") or 0)
    st.progress(min(1.0, max(0.0, score)))
    st.caption(f"Score {score:.2f} ({confidence.get('level', 'unknown')}), threshold {confidence.get('threshold')}")
    factors = confidence.get("factors", {})
    if factors:
        st.dataframe(
            pd.DataFrame([{"factor": name, "value": value} for name, value in factors.items()]),
            hide_index=True,
            width="stretch",
        )
    for reason in confidence.get("reasons", []):
        st.markdown(f"- {reason}")

with st.expander("Human escalation"):
    if escalation:
        st.json(escalation)
    else:
        st.caption("This ticket was not escalated. Escalation records are written to data/escalations.json.")
