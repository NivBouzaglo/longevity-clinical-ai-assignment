"""Retrieval over the guideline corpus (data/guidelines/) for citation-grounded answers.

Each guideline markdown file is chunked by its ``##`` section headers (`What it
estimates`, `Risk factors used`, `Modifiable levers`, `Interpreting the
estimate`) — one chunk per section, with the file's `#`-level title prepended so
a chunk reads well standalone. The corpus is tiny (5 files, ~4 sections each), so
whole-file chunking would be too coarse for good citation: section-level
chunking lets a query like "what's driving dementia risk" retrieve specifically
the "Risk factors used" section rather than the whole file.

Chunks are embedded with chromadb's default embedding function (ONNX-based, no
torch needed) and indexed via an in-memory ``chromadb.Client()`` — not persisted
to disk. Note this is a process-wide collection registry, not an isolated
instance per call (``create_collection`` errors if called twice with the same
name in one process); we use ``get_or_create_collection`` below for that
reason. The corpus is small and static, so the index is built once (lazily, on
first ``search()`` call, then cached) rather than persisted to disk — this
avoids any stale-index-vs-source-file drift or gitignore/rebuild-script
complexity.

This module is deliberately independent of FastMCP (no ``@mcp.tool`` here) so it
stays a plain, easily unit-testable function; ``mcp-server/server.py`` wires
``search`` up as the ``search_guidelines`` tool, importing it behind a
try/except since ``chromadb`` is an optional extra (``uv sync --extra rag``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import chromadb

REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDELINES_DIR = REPO_ROOT / "data" / "guidelines"

_TITLE_RE = re.compile(r"^# (.+)$", re.MULTILINE)
_SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)

# Module-level cache for the built collection — populated lazily by
# ``_get_collection()`` so the index is built once per process, not per query.
_collection: chromadb.api.models.Collection.Collection | None = None


def _chunk_file(path: Path) -> list[dict[str, str]]:
    """Split one guideline markdown file into one chunk per ``##`` section.

    Each chunk's text is the file's `#`-level title followed by that section's
    header and body, so the chunk is self-contained when returned out of
    context as a citation snippet.
    """
    text = path.read_text(encoding="utf-8")
    title_match = _TITLE_RE.search(text)
    title = title_match.group(1).strip() if title_match else path.stem

    headers = list(_SECTION_RE.finditer(text))
    chunks: list[dict[str, str]] = []
    for i, header in enumerate(headers):
        section = header.group(1).strip()
        start = header.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[start:end].strip()
        chunks.append(
            {"text": f"{title}\n\n## {section}\n{body}", "source": path.name, "section": section}
        )
    return chunks


def _build_index() -> chromadb.api.models.Collection.Collection:
    """Chunk every guideline file and index the chunks in a fresh in-memory collection."""
    client = chromadb.Client()  # in-memory — no on-disk persistence
    # get_or_create (not create_collection): defense-in-depth against the module
    # docstring's note above — safe even if _get_collection()'s cache is ever
    # bypassed and this runs more than once in the same process.
    collection = client.get_or_create_collection("guidelines")

    # README.md documents the corpus itself (how to build this RAG tool) — it's
    # not a clinical guideline. Indexing it would let internal build notes get
    # surfaced/cited as if they were clinical guidance.
    chunks = [
        chunk
        for path in sorted(GUIDELINES_DIR.glob("*.md"))
        if path.name != "README.md"
        for chunk in _chunk_file(path)
    ]
    if chunks:
        collection.add(
            ids=[f"{c['source']}::{c['section']}" for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"], "section": c["section"]} for c in chunks],
        )
    return collection


def _get_collection() -> chromadb.api.models.Collection.Collection:
    """Return the module-level index, building it on first use."""
    global _collection
    if _collection is None:
        _collection = _build_index()
    return _collection


def search(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Return the top-k guideline chunks most relevant to `query`.

    Each result dict has:
    - `text` — the chunk (guideline title + section header + body).
    - `source` — the guideline's filename (e.g. `dementia_caide.md`), so callers
      can cite where the snippet came from.
    - `section` — the `##` section header the chunk came from.
    - `distance` — chromadb's similarity distance for this match (lower is more
      relevant).
    """
    collection = _get_collection()
    result = collection.query(query_texts=[query], n_results=k)

    documents = result.get("documents") or [[]]
    metadatas = result.get("metadatas") or [[]]
    distances = result.get("distances") or [[]]

    return [
        {"text": doc, "source": meta["source"], "section": meta["section"], "distance": dist}
        for doc, meta, dist in zip(documents[0], metadatas[0], distances[0])
    ]
