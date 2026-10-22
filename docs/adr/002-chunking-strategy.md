# ADR 002: Use structure-aware chunking with sentence-boundary fallback

- **Status:** Accepted for the initial project architecture
- **Date:** 2026-10-22

## Context

Chunking determines what evidence is available to retrieval and generation. Chunks that are too large dilute semantic focus and consume context budget; chunks that are too small lose context and increase index size. Inputs include PDFs, DOCX, images/OCR, CSV, and web pages, which have different natural structures. The strategy must preserve provenance and allow later evaluation-driven tuning.

### Alternatives

1. **Fixed-size chunking:** splits text by a target token/character count, often with overlap. It is simple, fast, predictable, and useful as a fallback, but may split sentences, tables, code blocks, or logical sections.
2. **Sentence-boundary chunking:** groups complete sentences until a token target is reached. It preserves local readability and is relatively inexpensive, but sentences alone do not capture section hierarchy and may perform poorly for tables, lists, or noisy OCR.
3. **Semantic chunking:** groups content when semantic similarity changes, often using embeddings or a model. It can preserve topic coherence, but is more expensive, less deterministic, harder to tune, and can produce variable chunk sizes.

## Decision

Use **structure-aware, sentence-boundary chunking** as the default. Preserve document headings, paragraph and list boundaries, page boundaries, and table/CSV row groups when extractors expose them. Pack adjacent units up to a configurable token target (initial target approximately 500 tokens with approximately 75 tokens of overlap, subject to model/tokenizer testing). Keep source offsets and provenance on each chunk.

Special cases:
- **CSV:** keep headers with row groups; avoid splitting a row's fields across chunks.
- **Tables:** preserve captions/headers with the corresponding rows when feasible.
- **Code and identifiers:** preserve fenced/code blocks and exact tokens; do not normalize away punctuation.
- **OCR:** remove obvious extraction noise conservatively and retain page coordinates when available.
- **Long paragraphs:** split at sentence boundaries first, then use fixed-token splitting as a bounded fallback.
- **Tiny sections:** merge with a neighboring section only when provenance and topic remain coherent.

## Consequences

**Benefits**
- Better source traceability and more coherent context than naive fixed-size splitting.
- Deterministic, testable behavior with lower cost and variance than semantic chunking.
- Modality-specific handling reduces damage to tables and structured records.

**Costs and risks**
- Extractors must expose structural metadata consistently.
- Sentence detection can fail on abbreviations, multilingual content, OCR errors, and code.
- A single token target will not be optimal for all domains or models.
- Overlap can duplicate evidence and increase index size.

**Switch/tune conditions**
- Evaluate chunking profiles on a labeled query set using recall@k, nDCG@k, context precision, faithfulness, token cost, and latency.
- Switch a specific domain to semantic chunking only when it delivers a repeatable retrieval/answer-quality improvement large enough to justify embedding cost and operational complexity.
- Increase/decrease size or overlap when evaluation shows missing context, noisy context, excessive duplication, or poor context-window utilization.
- Keep a fixed-size fallback for malformed or structureless text. Version every chunking profile and record it on each document version so results remain reproducible.
