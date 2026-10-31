# Janus-Graph MCP Protocol & Integration Guide 🔌

**Janus-Graph** exposes a standardized **Model Context Protocol (MCP)** interface via `FastMCP` over `stdio` or `SSE`. This enables seamless integration with AI agent frameworks (PicoClaw, Claude Desktop, Cursor, LangChain, AutoGen, etc.).

---

## Available MCP Tools

### 1. `add_episode`
Ingest a text episode or conversation turn into the knowledge graph.
- **Parameters**:
  - `content` (`string`, required): Text content to ingest.
  - `name` (`string`, optional): Title or label for the episode.
  - `source_description` (`string`, optional): Provenance metadata (e.g. `'user_conversation'`, `'agent_scratchpad'`). Part of the dedup hash, so the same body under a different provenance is a DIFFERENT episode.
- **Returns**: Dictionary with `success: true`, `queued: true` and `episode_id: "<uuid>"` in `<500ms`.

  **Duplicate handling**: controlled by `pipeline.dedup.mode` (`off` | `observe` | `enforce`).
  - `off` (default) — no dedup at all; every call queues a new episode.
  - `observe` — the duplicate is detected and counted, but the episode is STILL queued.
  - `enforce` — the duplicate is NOT queued. The call returns `success: true`, `queued: false`, `duplicate: true`, and `existing_episode_id` (the id of the row that already holds this payload). It is a structured result, not a raised error.

  A "duplicate" means the same `(content, group_id, source_description)` triple as a row still in status `queued` or `processing`. Rows already `done`/`failed` do not block a new enqueue.

### 2. `search_memory`
Search the knowledge graph using hybrid semantic vector and graph retrieval.
- **Parameters**:
  - `query` (`string`, required): Natural language query or question.
  - `limit` (`int`, default: `5`): Maximum facts/edges to return.
  - `group_id` (`string`, optional): Target partition name.
- **Returns**: Ranked list of relevant relational facts and temporal context.

### 3. `get_entity`
Retrieve a specific entity node and all of its 1-hop inbound and outbound relationships.
- **Parameters**:
  - `entity_name` (`string`, required): Name of the entity (supports case-insensitive matching).
- **Returns**: Entity properties and edge relationships.

### 4. `list_entities`
List entities stored in the knowledge graph with relationship degree counts.
- **Parameters**:
  - `limit` (`int`, default: `50`): Number of entities to return.
  - `filter_query` (`string`, optional): Substring filter for entity names.

### 5. `cypher_query`
Execute a read-only openCypher query directly against FalkorDB.
- **Security**: Strict read-only gating rejects queries containing mutating keywords (`CREATE`, `SET`, `DELETE`, `DROP`, `MERGE`, `REMOVE`, `CALL`).
- **Parameters**:
  - `query` (`string`, required): Read-only Cypher query (e.g., `MATCH (n) RETURN count(n)`).

### 6. `queue_status`
Inspect the SQLite ingestion queue and Dead-Letter Queue (DLQ).
- **Parameters**:
  - `episode_id` (`string`, optional): Detailed record lookup for a specific episode.
  - `limit` (`int`, default: `20`): Number of recent DLQ records to return.

### 7. `retry_dlq_episode`
Manually replay a dead-letter episode.
- **Parameters**:
  - `episode_id` (`string`, required): UUID of the episode to reset to `queued`.

### 8. `run_dream_maintenance`
Trigger memory consolidation cycle (clustering, deduplication, orphan pruning, DLQ auto-repair).
- **Parameters**:
  - `force` (`bool`, default: `false`): Force re-clustering regardless of min episode thresholds.
  - `group_id` (`string`, optional): Target partition group.

### 9. `graphiti_health`
Return a composite health snapshot of FalkorDB, SQLite WAL queue, embeddings, and DLQ status.

---

## Client Integration Configuration

### Claude Desktop Configuration (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "janus-graph": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/janus-graph",
        "run",
        "janus-graph",
        "mcp"
      ]
    }
  }
}
```
