"""Knowledge-base ingestion and persistent ChromaDB storage.

Markdown policy files (and any PDFs dropped into the folder) are read, split
into sections and chunks, embedded with a small deterministic local embedder
and stored in a persistent Chroma collection. A manifest signature means the
index is only rebuilt when the documents, chunking or embedder change.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import chromadb
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings, get_settings

COLLECTION_NAME = "resolveai_kb"
EMBEDDER_NAME = "resolveai-local-hash-v1"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
EMBEDDING_DIMENSIONS = 512

DOCUMENT_META = {
    "refund_policy.md": ("Refund and Return Policy", "refunds_returns"),
    "delivery_policy.md": ("Delivery Policy", "orders_delivery"),
    "subscription_policy.md": ("Subscription Policy", "subscriptions"),
    "troubleshooting.md": ("Technical Troubleshooting and Account Access", "technical"),
}


class LocalHashEmbedder:
    """Deterministic offline embedding function.

    Token and token-pair features are hashed into a fixed-size vector and
    normalised. It is a lexical embedding, not a neural one: good enough for a
    small policy knowledge base and for running without API keys or model
    downloads, and clearly reported as such.
    """

    def name(self) -> str:
        return EMBEDDER_NAME

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIMENSIONS
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.md5(feature.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % EMBEDDING_DIMENSIONS
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def __call__(self, input):  # noqa: A002 - Chroma's legacy signature
        return self.embed_documents(input)

    def embed_documents(self, input):  # noqa: A002 - Chroma's current signature
        return [self._embed(text) for text in input]

    def embed_query(self, input):  # noqa: A002 - Chroma's current signature
        items = [input] if isinstance(input, str) else list(input)
        vectors = [self._embed(text) for text in items]
        return vectors[0] if isinstance(input, str) else vectors


def _markdown_sections(path: Path) -> list[dict]:
    """Split a markdown policy into its `##` sections."""
    title, category = DOCUMENT_META.get(path.name, (path.stem.replace("_", " ").title(), "general"))
    lines = path.read_text(encoding="utf-8").splitlines()
    sections: list[dict] = []
    current = "Overview"
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append({"section": current, "text": text})

    for line in lines:
        if line.startswith("## "):
            flush()
            buffer = []
            current = line[3:].strip()
        elif not line.startswith("# "):
            buffer.append(line)
    flush()
    return [{"title": title, "category": category, "source": path.name, **section} for section in sections]


def _pdf_sections(path: Path) -> list[dict]:
    """Extract PDF text page by page, keeping the page number as metadata."""
    title = path.stem.replace("_", " ").title()
    pages = []
    with pymupdf.open(path) as document:
        for number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"title": title, "category": "general", "source": path.name, "section": f"Page {number}", "page": number, "text": text})
    return pages


def load_sections(settings: Settings | None = None) -> list[dict]:
    """Read every supported document in the knowledge-base folder."""
    settings = settings or get_settings()
    folder = settings.knowledge_base_path
    sections: list[dict] = []
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() in {".md", ".markdown"}:
            sections.extend(_markdown_sections(path))
        elif path.suffix.lower() == ".pdf":
            sections.extend(_pdf_sections(path))
    return sections


def build_chunks(settings: Settings | None = None) -> list[dict]:
    """Turn sections into overlapping chunks with stable IDs and metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: list[dict] = []
    for section in load_sections(settings):
        body = f"{section['title']} - {section['section']}\n{section['text']}"
        for position, piece in enumerate(splitter.split_text(body)):
            chunks.append(
                {
                    "id": f"{section['source']}::{section['section']}::{position}",
                    "text": piece,
                    "metadata": {
                        "source": section["source"],
                        "title": section["title"],
                        "section": section["section"],
                        "category": section["category"],
                        "page": int(section.get("page") or 0),
                    },
                }
            )
    return chunks


def signature(settings: Settings | None = None) -> str:
    """Fingerprint of documents, chunking and embedder."""
    settings = settings or get_settings()
    digest = hashlib.sha256()
    for path in sorted(settings.knowledge_base_path.glob("*")):
        if path.suffix.lower() in {".md", ".markdown", ".pdf"}:
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    digest.update(f"{CHUNK_SIZE}/{CHUNK_OVERLAP}/{EMBEDDER_NAME}".encode("utf-8"))
    return digest.hexdigest()[:16]


def _manifest_path(settings: Settings) -> Path:
    return settings.chroma_db_path / "ingest_manifest.json"


def _read_manifest(settings: Settings) -> dict:
    try:
        return json.loads(_manifest_path(settings).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_manifest(settings: Settings, payload: dict) -> None:
    settings.chroma_db_path.mkdir(parents=True, exist_ok=True)
    _manifest_path(settings).write_text(json.dumps(payload, indent=2), encoding="utf-8")


@lru_cache(maxsize=2)
def _open_collection(path_str: str, _signature: str):
    client = chromadb.PersistentClient(path=path_str)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=LocalHashEmbedder(),
        metadata={"hnsw:space": "cosine"},
    )


def get_collection(settings: Settings | None = None):
    """Return the ready-to-query Chroma collection."""
    settings = settings or get_settings()
    ensure_index(settings)
    return _open_collection(str(settings.chroma_db_path), signature(settings))


def build_index(force: bool = False, settings: Settings | None = None) -> dict:
    """Build or refresh the index; returns a small status dictionary."""
    settings = settings or get_settings()
    settings.chroma_db_path.mkdir(parents=True, exist_ok=True)
    fingerprint = signature(settings)
    manifest = _read_manifest(settings)

    client = chromadb.PersistentClient(path=str(settings.chroma_db_path))
    try:
        existing = client.get_collection(COLLECTION_NAME, embedding_function=LocalHashEmbedder())
        existing_count = existing.count()
    except Exception:
        existing_count = 0

    if not force and manifest.get("signature") == fingerprint and existing_count > 0:
        return {
            "status": "up_to_date",
            "signature": fingerprint,
            "counts": {"chunks": existing_count, "documents": manifest.get("documents", 0)},
            "built_at": manifest.get("built_at"),
        }

    chunks = build_chunks(settings)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=LocalHashEmbedder(),
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    documents = len({chunk["metadata"]["source"] for chunk in chunks})
    built_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    payload = {
        "signature": fingerprint,
        "built_at": built_at,
        "chunks": len(chunks),
        "documents": documents,
        "embedder": EMBEDDER_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    _write_manifest(settings, payload)
    _open_collection.cache_clear()
    return {"status": "rebuilt", "signature": fingerprint, "counts": {"chunks": len(chunks), "documents": documents}, "built_at": built_at}


def ensure_index(settings: Settings | None = None) -> dict:
    """Build the index on first use, otherwise leave it untouched."""
    return build_index(force=False, settings=settings)


def index_status(settings: Settings | None = None) -> dict:
    """Report whether the persistent index matches the current documents."""
    settings = settings or get_settings()
    manifest = _read_manifest(settings)
    current = signature(settings)
    try:
        collection = chromadb.PersistentClient(path=str(settings.chroma_db_path)).get_collection(
            COLLECTION_NAME, embedding_function=LocalHashEmbedder()
        )
        count = collection.count()
    except Exception:
        count = 0
    return {
        "exists": count > 0,
        "up_to_date": bool(count) and manifest.get("signature") == current,
        "chunks": count,
        "documents": manifest.get("documents", 0),
        "built_at": manifest.get("built_at"),
        "embedder": manifest.get("embedder", EMBEDDER_NAME),
        "signature": manifest.get("signature"),
    }
