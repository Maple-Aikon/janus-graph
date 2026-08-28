# Janus-Graph Architecture 🏛️

**Janus-Graph** is an enterprise-grade, modular, and resilient Knowledge Graph Memory Engine specifically engineered for autonomous AI agents and complex agentic workflows.

It unifies fast embedded graph traversal via **FalkorDB** with dynamic semantic memory and temporal knowledge extraction via **Graphiti Core**, bound together through an asynchronous, transactionally durable **SQLite WAL Pipeline** and self-repairing **LLM Schema Heuristics**.

---

## High-Level System Architecture

```text
+--------------------------------------------------------------------------------+
|                                 Agent Ecosystem                                |
|             (PicoClaw, Claude Desktop, Custom LLM Frameworks, CLI)              |
+--------------------------------------------------------------------------------+
                                        │
                       FastMCP Protocol / JSON-RPC / CLI
                                        ▼
+────────────────────────────────────────────────────────────────────────────────+
|                            Janus-Graph Unified Engine                          |
|                                                                                |
|  +─────────────────────────+                     +──────────────────────────+  |
|  |      MCP Interface      |                     |        Unified CLI       |  |
|  |   (FastMCP stdio/SSE)   |                     |   (Doctor, Sweep, Dream) |  |
|  +────────────┬────────────+                     +────────────┬─────────────+  |
|               │                                               │                |
|               └──────────────────────┬────────────────────────┘                |
|                                      ▼                                         |
|  +──────────────────────────────────────────────────────────────────────────+  |
|  |                        Resilient Ingestion Pipeline                      |  |
|  |                                                                          |  |
|  |   +───────────────────+     Async Queue     +────────────────────────+   |  |
|  |   | Ingestion Endpoint| ──────────────────> |   SQLite WAL Database  |   |  |
|  |   |  (<500ms return)  |                     | (FIFO, Retry, Backoff) |   |  |
|  |   +───────────────────+                     +───────────┬────────────+   |  |
|  |                                                         │                |  |
|  |                                                         ▼                |  |
|  |   +───────────────────+                     +────────────────────────+   |  |
|  |   |  Dead-Letter-Queue| <── (Max Retries) ──|   Cron Worker Engine   |   |  |
|  |   |       (DLQ)       |                     | (Semaphore Concurrency)|   |  |
|  |   +───────────────────+                     +───────────┬────────────+   |  |
|  +─────────────────────────────────────────────────────────┼────────────────+  |
|                                                            │                   |
|                                                            ▼                   |
|  +──────────────────────────────────────────────────────────────────────────+  |
|  |                      Schema-Repairing Heuristics Client                  |  |
|  |                                                                          |  |
|  |     [ LLM Extraction Payload (LiteLLM / OpenAI / Anthropic / Gemini) ]     |  |
|  |                                     │                                    |  |
|  |                                     ▼ (Pydantic ValidationError)         |  |
|  |     [ Heuristics Rule Engine: EdgeDuplicate, ExtractedEntities, etc. ]   |  |
|  |                                     │                                    |  |
|  |                                     ▼                                    |  |
|  |         [ Quirks Logger: /data/logs/llm_schema_quirks.jsonl ]            |  |
|  +─────────────────────────────────────┬────────────────────────────────────+  |
|                                        │                                       |
|                                        ▼                                       |
|  +──────────────────────────────────────────────────────────────────────────+  |
|  |                     Graphiti Semantic Extraction Core                    |  |
|  |        (Entity Deduplication, Temporal Edge Extraction, Search)          |  |
|  +─────────────────────────────────────┬────────────────────────────────────+  |
|                                        │                                       |
|                                        ▼                                       |
|  +──────────────────────────────────────────────────────────────────────────+  |
|  |                       FalkorDB Graph Engine Wrapper                      |  |
|  |                                                                          |  |
|  |   - Cypher Query Execution & Graph Traversal                             |  |
|  |   - Redis-server daemon management (Unix socket / TCP port 6379)         |  |
|  |   - Vector Embeddings Storage (Llama.cpp / OpenAI / HuggingFace)         |  |
|  +──────────────────────────────────────────────────────────────────────────+  |
+────────────────────────────────────────────────────────────────────────────────+
```

---

## Subsystems & Modular Components

The codebase is organized into independent, decoupled packages following Clean Architecture principles:

### 1. `janus_graph.config`
Centralized configuration management powered by `pydantic-settings`.
- Reads configuration from YAML (`janus_graph.yaml` or `config.yaml`) with full fallback to environment variables (`.env`).
- Resolves dynamic variables (`${DATA_DIR}`, `${HOME}`) and auto-creates necessary data, socket, and logging directories.
- Sub-configs: `DatabaseConfig`, `ServerConfig`, `PipelineConfig`, `DreamConfig`, `LLMConfig`, `EmbeddingConfig`, `ReportingConfig`.

### 2. `janus_graph.core`
Contract interfaces, abstract base classes, and thread-safe singleton managers.
- `contracts.py`: Defines abstract protocols (`IGraphStore`, `IEpisodeQueue`, `ISchemaRepairRule`, `IQuirksLogger`, `ICacheBackend`).
- `instance.py`: Manages lifecycle, client caching, and unified access to initialized Graphiti and FalkorDB instances.

### 3. `janus_graph.engine`
Daemon manager and connection client for FalkorDB.
- `server.py`: Manages the embedded Redis/FalkorDB binary lifecycle (start, stop, pid checking, unix socket / TCP binding, config generation).
- `client.py`: High-level wrapper for FalkorDB connections, read-only Cypher security gating, node/edge lookups, and graph topology queries.

### 4. `janus_graph.pipeline`
Resilient asynchronous ingestion, queue workers, and background maintenance.
- `queue.py`: SQLite WAL-backed queue with atomic transaction management, state transitions (`queued` -> `processing` -> `done` | `aborted` | `failed`), and Dead-Letter Queue (DLQ) isolation.
- `worker.py`: Semaphore-bounded concurrent episode processor invoking Graphiti extraction.
- `cron.py`: Batch sweep runner, SIGINT/SIGTERM graceful shutdown handler, and multi-channel health notifier.
- `dream.py`: 4-phase nightly consolidation routine:
  1. Community clustering via Leiden / modularity graph algorithms.
  2. 2-Tier semantic entity deduplication.
  3. True orphan node pruning.
  4. Idempotent DLQ automated re-evaluation.

### 5. `janus_graph.heuristics`
Self-healing LLM schema validation and anomaly logging.
- `repairing_client.py`: `SchemaRepairingLLMClient` intercepting Pydantic `ValidationError` from LLM output, extracting raw input structures, executing applicable heuristic rules, and reconstructing valid payloads without incurring costly LLM re-queries.
- `registry.py`: Dynamic rule registry supporting rule priorities and runtime plugin registration.
- `rules/`: Built-in modular repair rules (`EdgeDuplicateRule`, `ExtractedEntitiesRule`, `ExtractedEdgesRule`, `NodeResolutionsRule`).
- `quirks_logger.py`: Structured JSONL telemetry recorder logging schema anomalies per model, provider, and error signature.

### 6. `janus_graph.mcp`
FastMCP server standardizing access for AI agent frameworks over `stdio` or `SSE`.
- Exposes tools for episode ingestion (`add_episode`), semantic search (`search_memory`), entity exploration (`get_entity`, `list_entities`), read-only Cypher querying (`cypher_query`), queue management (`queue_status`, `retry_dlq_episode`), and maintenance (`run_dream_maintenance`, `graphiti_health`).

### 7. `janus_graph.cache`
Multi-tiered embedding and query caching layer.
- `lru.py`: In-memory thread-safe LRU cache.
- `embed_cache.py`: Persistent or cached vector representations to minimize redundant embedding API calls.

### 8. `janus_graph.cli`
Unified CLI tooling for operators and automation.
- Subcommands: `doctor`, `engine`, `sweep`, `dream`, `mcp`, `snapshot`, `rollback`.

---

## Data Flow Lifecycles

### 1. Ingestion Flow (< 500ms Latency)
1. Agent invokes `add_episode(content, name, source_description)` via MCP or Python API.
2. Ingestion endpoint writes the raw episode into SQLite WAL table with status `queued`.
3. Endpoint returns the generated UUID immediately to caller.
4. Background cron worker picks up `queued` episodes in batches, acquires concurrency semaphores, and processes them through Graphiti extraction.

### 2. LLM Extraction & Heuristic Self-Repair
1. Graphiti prompts LLM with schema contracts (entities, edges, temporal timestamps).
2. If the LLM provider (e.g. LiteLLM, Ollama) returns non-conforming JSON (e.g., missing default lists, stringified integers):
   - Pydantic raises `ValidationError`.
   - `SchemaRepairingLLMClient` intercepts the exception.
   - Matching heuristic rules mutate the payload in-memory into schema compliance.
   - Payload is re-validated and returned cleanly to Graphiti.
   - Anomaly is appended to `llm_schema_quirks.jsonl` for offline evaluation.

### 3. Query & Recall Flow
1. Agent calls `search_memory(query, limit)`.
2. Query text is embedded (with local/remote embedding model, checked against cache).
3. FalkorDB executes vector cosine similarity across entity/edge embeddings.
4. Graphiti traverses temporal and relational links, returning contextual facts and confidence scores.
