"""Unit tests for guidelines_retrieval.py — chunking + chromadb retrieval.

These exercise the retrieval module directly, in-process, with no live MCP
server needed. `chromadb` is a required optional-extra dependency here (`uv
sync --extra rag`) — no mocking of it, since the corpus is tiny and the
in-memory index builds fast.

Run:  uv run pytest mcp-server/tests/test_guidelines_retrieval.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from guidelines_retrieval import GUIDELINES_DIR, _chunk_file, _get_collection, search

DEMENTIA_FILE = GUIDELINES_DIR / "dementia_caide.md"

# Non-README guideline files actually present in data/guidelines/, verified
# against the real directory rather than assumed.
GUIDELINE_FILES = sorted(
    p for p in GUIDELINES_DIR.glob("*.md") if p.name != "README.md"
)


# ---------------------------------------------------------------------------
# _chunk_file — happy flow
# ---------------------------------------------------------------------------


def test_chunk_file_produces_one_chunk_per_section() -> None:
    """dementia_caide.md has 4 `##` sections -> 4 chunks."""
    chunks = _chunk_file(DEMENTIA_FILE)
    assert len(chunks) == 4


def test_chunk_file_sources_and_sections_are_correct() -> None:
    chunks = _chunk_file(DEMENTIA_FILE)
    expected_sections = [
        "What it estimates",
        "Risk factors used",
        "Modifiable levers",
        "Interpreting the estimate",
    ]
    assert [c["section"] for c in chunks] == expected_sections
    assert all(c["source"] == "dementia_caide.md" for c in chunks)


def test_chunk_file_text_starts_with_title_and_contains_section_header() -> None:
    chunks = _chunk_file(DEMENTIA_FILE)
    for chunk in chunks:
        assert chunk["text"].startswith("Dementia Risk — CAIDE score (summary)")
        assert f"## {chunk['section']}" in chunk["text"]


# ---------------------------------------------------------------------------
# Regular workflow — chunking each real guideline file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", GUIDELINE_FILES, ids=lambda p: p.name)
def test_chunk_file_every_guideline_file_has_four_sections(path: Path) -> None:
    """Every real (non-README) guideline file chunks into exactly 4 sections."""
    chunks = _chunk_file(path)
    assert len(chunks) == 4
    assert all(c["source"] == path.name for c in chunks)


# ---------------------------------------------------------------------------
# Regression test — README.md must never be indexed
# ---------------------------------------------------------------------------


def test_build_index_excludes_readme_and_has_expected_total_chunk_count() -> None:
    """5 real guideline files x 4 sections each = 20 chunks; README.md excluded.

    This is a regression test for a bug where README.md (which documents the
    corpus itself, not a clinical guideline) was being chunked and indexed
    alongside the real guideline files.

    Goes through `_get_collection()` (the same lazily-cached entry point
    `search()` uses) rather than calling `_build_index()` directly: chromadb's
    "ephemeral" `Client()` shares a process-wide collection registry in this
    version, so building a second `guidelines` collection directly in the same
    process as `search()`'s cached one raises "Collection already exists".
    Production never does this (the module-level cache guards it), so tests
    shouldn't either.
    """
    # Verify the arithmetic against the actual directory contents rather than
    # hardcoding it blind.
    assert len(GUIDELINE_FILES) == 5
    expected_total = sum(len(_chunk_file(p)) for p in GUIDELINE_FILES)
    assert expected_total == 20

    collection = _get_collection()
    result = collection.get()

    assert collection.count() == expected_total
    assert collection.count() == 20
    sources = {meta["source"] for meta in result["metadatas"]}
    assert "README.md" not in sources
    assert sources == {p.name for p in GUIDELINE_FILES}


# ---------------------------------------------------------------------------
# search() — happy path relevance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected_source"),
    [
        ("What drives dementia risk?", "dementia_caide.md"),
        ("How is chronic kidney disease risk estimated?", "ckd_framingham.md"),
        ("What can be done to lower cardiovascular risk?", "cvd_prevent.md"),
    ],
)
def test_search_surfaces_expected_source_as_top_result(
    query: str, expected_source: str
) -> None:
    results = search(query, k=3)
    sources = [r["source"] for r in results]
    assert expected_source in sources


# ---------------------------------------------------------------------------
# search() — edge cases
# ---------------------------------------------------------------------------


def test_search_k_equals_1_returns_exactly_one_result() -> None:
    results = search("cardiovascular risk", k=1)
    assert len(results) == 1


def test_search_k_larger_than_corpus_does_not_crash() -> None:
    """k=100 far exceeds the 20-chunk corpus; chromadb caps results, no crash."""
    results = search("risk factors", k=100)
    assert 0 < len(results) <= 20


def test_search_empty_query_does_not_crash() -> None:
    results = search("", k=3)
    assert isinstance(results, list)


def test_search_result_dicts_have_expected_keys() -> None:
    results = search("blood pressure", k=3)
    assert results
    for r in results:
        assert set(r.keys()) == {"text", "source", "section", "distance"}
