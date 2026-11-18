"""Semantic retrieval over the persistent knowledge base.

Vector similarity from Chroma is blended with a small lexical-overlap score,
which keeps exact policy terms (``30 days``, ``past_due``) from being washed
out by paraphrases. Category metadata narrows the search when the ticket
category maps cleanly onto a policy document.
"""

from __future__ import annotations

import math
import re

from config import Settings
from knowledge_base import get_collection

CATEGORY_FILTERS = {
    "refunds_returns": "refunds_returns",
    "orders_delivery": "orders_delivery",
    "subscription": "subscriptions",
    "technical": "technical",
}


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 2}


def _lexical_overlap(query: str, text: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & _tokens(text))
    return overlap / math.sqrt(len(query_tokens) * 4)


def _normalise(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def source_label(chunk: dict) -> str:
    """Human-readable citation label for a retrieved chunk."""
    page = chunk.get("page") or 0
    if page:
        return f"{chunk['source']} p. {page}"
    section = chunk.get("section")
    return f"{chunk['source']} - {section}" if section else chunk["source"]


def search(query: str, k: int = 4, category: str | None = None, settings: Settings | None = None) -> list[dict]:
    """Return the best matching policy chunks for a query."""
    where = None
    mapped = CATEGORY_FILTERS.get((category or "").lower())
    if mapped:
        where = {"category": mapped}

    collection = get_collection(settings)
    result = collection.query(query_texts=[query], n_results=max(k * 3, k), where=where)
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]

    candidates = []
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        semantic = max(0.0, min(1.0, 1.0 - float(distance or 0.0)))
        candidates.append(
            {
                "id": chunk_id,
                "text": text,
                "source": metadata.get("source", ""),
                "title": metadata.get("title", ""),
                "section": metadata.get("section", ""),
                "category": metadata.get("category", ""),
                "page": metadata.get("page", 0),
                "semantic": semantic,
                "lexical": _lexical_overlap(query, text),
            }
        )

    normalised = _normalise([candidate["semantic"] for candidate in candidates])
    for candidate, semantic_score in zip(candidates, normalised):
        candidate["score"] = round(0.6 * semantic_score + 0.4 * candidate["lexical"], 4)

    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)[:k]
    for chunk in ranked:
        chunk["label"] = source_label(chunk)
    return ranked
