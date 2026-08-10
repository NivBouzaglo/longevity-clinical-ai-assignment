# Guideline Corpus (bonus retrieval surface)

A tiny corpus of **simplified, educational** summaries — one per risk model —
describing what each instrument estimates and which factors drive it.

> ⚠️ These are paraphrased summaries written for this exercise, **not** verbatim
> or authoritative clinical guidelines. They exist so the assistant can ground and
> **cite** its risk explanations in retrievable text.

| File | Risk | Instrument |
|---|---|---|
| `cvd_prevent.md` | CVD | AHA PREVENT |
| `t2dm_ada.md` | T2DM | ADA Diabetes Risk Test |
| `ckd_framingham.md` | CKD | Framingham CKD score |
| `cld_clivd.md` | CLD | CLivD score |
| `dementia_caide.md` | Dementia | CAIDE score |

## Intended use (bonus RAG task)
Build a `search_guidelines(query, k)` MCP tool that:
1. Chunks and embeds these documents into a lightweight vector store
   (`uv sync --extra rag` installs `chromadb`; its default embedder is ONNX-based,
   no GPU/torch needed — or use any embedding model you prefer).
2. Returns the top-k relevant snippets **with their source filename** so the
   assistant can cite where an explanation came from.

This lets you demonstrate embeddings, retrieval, chunking, and
citation-faithfulness — and pairs naturally with the `citation` eval cases.
