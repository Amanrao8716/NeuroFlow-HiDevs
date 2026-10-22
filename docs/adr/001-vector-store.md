# ADR 001: Use PostgreSQL with pgvector as the vector store

- **Status:** Accepted for the initial project architecture
- **Date:** 2026-10-22
- **Decision owners:** NeuroFlow project

## Context

NeuroFlow requires dense-vector similarity search, keyword search, metadata filtering, tenant isolation, and durable relationships between documents, versions, chunks, queries, and evaluations. The project also needs to keep infrastructure manageable while the system is built incrementally. The vector store must support reproducible metadata joins and an operational path that can be developed locally.

Alternatives considered:

- **Pinecone:** managed vector database with reduced operational burden, but adds a separate system and service-specific dependency/cost.
- **Weaviate:** capable vector database with hybrid search and schema features, but adds another service and its own operational/data-model conventions.
- **Qdrant:** strong vector search and filtering, but also introduces a separate database and consistency boundary for chunk metadata.
- **PostgreSQL + pgvector:** combines relational metadata, transactions, access-control joins, full-text search, and vector storage in one system.

## Decision

Use PostgreSQL as the system of record and pgvector for embeddings in the initial implementation. Use PostgreSQL full-text search for the lexical retrieval branch and combine its ranked results with dense retrieval through Reciprocal Rank Fusion. Store document/chunk metadata and vector references together so tenant/ACL filters can be enforced in the same query path.

Use a vector index and distance operator appropriate to the selected embedding model and workload. Benchmark exact search first on small datasets, then select HNSW or IVFFlat based on latency, recall, memory, build time, and update characteristics. Keep embedding dimensions/model profiles explicit; do not compare incompatible embeddings in the same index.

## Consequences

**Benefits**
- Fewer services and a simpler local development story.
- Relational joins and transactions help preserve document-version/chunk consistency.
- Metadata, tenant filters, lexical search, and vector search can be coordinated in one database.
- Lower integration overhead for evaluation and traceability records.

**Costs and risks**
- PostgreSQL becomes a critical shared dependency and needs backups, monitoring, capacity planning, and migration discipline.
- Very large vector workloads or high write/search contention may need a dedicated vector service.
- Index choice and filter selectivity can materially affect recall and latency; performance must be measured rather than assumed.
- Search features may not match the specialized scaling characteristics of every managed vector database.

**Revisit when:** representative load tests show that PostgreSQL cannot meet agreed latency/recall/cost targets after query and index tuning, or when independent vector scaling is operationally necessary. Any migration must preserve chunk IDs, metadata filters, ranking traces, and evaluation comparability.
