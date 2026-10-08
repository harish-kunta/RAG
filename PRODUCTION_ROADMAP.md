# From learning demo to production RAG chat

## What the project has today

The existing pipeline reads Markdown/text files, SQLite rows, and a local cache of Notion pages. It chunks the text, creates embeddings, saves a small JSON vector index, supports lexical/semantic/hybrid retrieval, packs passages into a bounded context, and asks an OpenAI model to answer with source citations. The web milestone adds a browser chat and an HTTP endpoint around that pipeline.

## What is still missing for production

This app is a local, single-user learning tool. The browser holds the current chat in memory and sends a short history with each request. There is no account system, saved conversation database, team separation, or authorization boundary. The JSON vector index is convenient for learning, but it is not a concurrent, scalable vector store. SQLite and Notion are ingested by the existing local connector/cache steps; they are not continuously synchronized. Requests are synchronous and there is no queue, streaming response, quota, abuse protection, alerting, backup, or deployment setup.

Do not expose this version directly to the public internet. Metadata filters and citations improve relevance and explainability; they are not access control. If sources contain private data, production retrieval must enforce the caller's permissions before any passages reach the model or browser.

## A practical hardening sequence

### 1. Local chat foundation — current milestone

- Serve a browser UI and API from one origin.
- Reuse the existing retrieval functions and return the actual passages used as sources.
- Keep the model key on the server in an environment variable or secret manager.
- Keep chat history in the browser for now; no user accounts or durable chat storage.

### 2. Durable application data

- Move chat users, conversations, messages, documents, and ingestion status into PostgreSQL with migrations.
- Use PostgreSQL with `pgvector` for a modest first deployment, or a managed vector database when scale/operations justify it.
- Keep original uploaded files in private object storage; store checksums, owners, timestamps, status, and access labels in PostgreSQL.
- Add retention and deletion flows, including removal of vectors derived from deleted sources.

### 3. Reliable ingestion

- Add authenticated upload and connector setup flows with file-type/size limits.
- Run parsing, chunking, embedding, and indexing as background jobs; return job status in the UI.
- Make jobs retryable and idempotent, record failures, and update document versions atomically so users never search a half-built index.
- Schedule incremental connector syncs and detect deletions/permission changes, not just new or edited pages.
- Add parsers for the file formats the product promises to support, plus malware scanning if arbitrary uploads are accepted.

### 4. Identity and data isolation

- Add a trusted identity provider and server-side sessions or verified tokens.
- Record tenant/user ownership on every conversation and source.
- Apply row-level authorization to source selection before retrieval; include permission changes in cache/index invalidation.
- Test cross-user access explicitly. Never rely on a UI filter or a model instruction to protect private records.

### 5. Retrieval and answer quality

- Grow the labeled evaluation set with real questions, expected sources, and answer expectations.
- Track retrieval measures (Recall@k/MRR) alongside answer correctness, citation support, and “not found” behavior.
- Compare chunking, metadata filters, hybrid retrieval, rerankers, and context budgets against that set before changing defaults.
- Add conversational query rewriting for follow-up questions and a reranker only when evaluation shows a benefit.
- Use token-based budgets, deduplicate near-identical passages, and inspect whether every generated citation maps to a returned source.

### 6. Safe, dependable chat API

- Add streaming with cancellation, per-user rate limits, request-size limits, model timeouts, retries for transient provider errors, and cost limits.
- Treat prompts, retrieved text, uploaded content, and connector output as untrusted. Keep tools/actions disabled unless explicitly authorized and separately guarded.
- Add moderation or domain-specific safety checks where the product needs them.
- Return stable error codes; do not return stack traces, provider payloads, or secrets to the browser.

### 7. Operations and release

- Add structured logs, request IDs, latency/error/cost metrics, traces across retrieval and generation, and alerts.
- Add Docker packaging, CI checks, dependency/security scanning, configuration validation, and staging deployment.
- Use managed secret storage, TLS, backups, restore exercises, database migrations, retention policy, and an incident runbook.
- Load-test ingestion and chat, set capacity limits, and monitor actual cost and answer quality after release.

## Suggested architecture after the local milestone

```text
Browser UI → authenticated API → conversation store (PostgreSQL)
                         ├── permission-checked retrieval → pgvector + keyword index
                         ├── background ingestion queue → parsers/embeddings → index
                         └── model provider → streamed answer + source references
Private object storage holds originals; logs/metrics record latency, failures, and cost.
```

## First production-shaped next step

Persist conversations and document metadata in PostgreSQL, then move vectors from the JSON file to `pgvector`. Keep the existing `Document → Chunk → retrieval → context → answer` interfaces while replacing the storage adapters. This preserves the concepts already learned while making it possible to add multiple users and background ingestion cleanly.
