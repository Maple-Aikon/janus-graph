# Janus-Graph Configuration Guide ⚙️

`janus-graph` uses a unified configuration engine powered by `pydantic-settings`. It supports hierarchical YAML files (`janus_graph.yaml` or `config.yaml`) with seamless overrides via environment variables (`.env` or shell env).

---

## Configuration File Structure

An example configuration file is provided at `config.example.yaml`:

```yaml
# ====================================================================
# Janus-Graph Unified Configuration
# ====================================================================

server:
  host: "0.0.0.0"
  port: 8000
  socket_path: "${DATA_DIR}/run/falkordb.sock"
  falkordb_port: 6379
  falkordb_bin: "bin/redis-server-8"
  falkordb_module: "bin/falkordb.so"
  falkordb_config: "data/falkordb.conf"

database:
  data_dir: "./data"
  graph_name: "janus_memory"
  episodes_db_path: "${DATA_DIR}/episodes.db"
  wal_mode: true
  busy_timeout_ms: 10000

pipeline:
  async_mode: true
  batch_size: 10
  concurrency_limit: 4
  max_attempts: 3
  retry_delay_base: 2.0
  retry_delay_max: 60.0
  reaper_interval_seconds: 300
  lease_duration_seconds: 600

dream:
  enabled: true
  cron_expression: "30 2 * * *"  # 02:30 AM daily
  timeout_seconds: 600
  min_episodes_threshold: 10
  dedup_similarity_threshold: 0.88
  prune_orphan_nodes: true
  auto_repair_dlq: true

llm:
  provider: "openai_generic"
  base_url: "http://127.0.0.1:4000/v1"
  api_key: "sk-placeholder"
  model: "gpt-4o-mini"
  temperature: 0.1
  max_tokens: 4096
  enable_heuristics_repair: true
  quirks_log_path: "${DATA_DIR}/logs/llm_schema_quirks.jsonl"

embedding:
  provider: "openai_generic"
  base_url: "http://127.0.0.1:8081/v1"
  api_key: "placeholder"
  model: "nomic-embed-text-v2"
  dimensions: 768
  cache_enabled: true
  cache_ttl_seconds: 86400

reporting:
  notify_telegram: false
  telegram_bot_token: ""
  telegram_chat_id: ""
  notify_webhook_url: ""
```

---

## Environment Variable Mappings

All settings can be overridden using environment variables prefixed with `JANUS_` or the direct legacy environment variables:

| Setting Path | Environment Variable | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `database.data_dir` | `JANUS_DATA_DIR` | `./data` | Root directory for runtime state, databases, and logs |
| `database.graph_name` | `JANUS_GRAPH_NAME` / `GRAPHITI_GROUP_ID` | `janus_memory` | Primary graph identifier partition |
| `database.episodes_db_path`| `JANUS_EPISODES_DB_PATH` | `${DATA_DIR}/episodes.db` | SQLite WAL queue database path |
| `server.falkordb_port` | `JANUS_FALKORDB_PORT` / `FALKORDB_PORT` | `6379` | TCP port for embedded FalkorDB |
| `server.socket_path` | `JANUS_SOCKET_PATH` | `${DATA_DIR}/run/falkordb.sock` | Unix domain socket path for low-latency IPC |
| `pipeline.async_mode` | `JANUS_ASYNC_MODE` / `ASYNC_MODE` | `true` | When true, `add_episode` queues asynchronously |
| `pipeline.concurrency_limit`| `JANUS_CONCURRENCY_LIMIT` | `4` | Worker parallel task semaphore bound |
| `pipeline.batch_size` | `JANUS_BATCH_SIZE` | `10` | Max episodes processed per sweep cycle |
| `llm.base_url` | `JANUS_LLM_BASE_URL` / `OPENAI_BASE_URL` | `http://127.0.0.1:4000/v1` | LLM Proxy or direct OpenAI API endpoint |
| `llm.api_key` | `JANUS_LLM_API_KEY` / `OPENAI_API_KEY` | `sk-placeholder` | LLM API authentication key |
| `llm.model` | `JANUS_LLM_MODEL` / `OPENAI_MODEL` | `gpt-4o-mini` | Model identifier for entity/edge extraction |
| `llm.enable_heuristics_repair` | `JANUS_ENABLE_HEURISTICS_REPAIR` | `true` | Intercept and repair invalid LLM JSON payloads |
| `embedding.base_url` | `JANUS_EMBEDDING_BASE_URL` | `http://127.0.0.1:8081/v1` | Embedding server endpoint (e.g., llama-server) |
| `embedding.dimensions`| `JANUS_EMBEDDING_DIMENSIONS` | `768` | Vector embedding dimension size |

---

## Variable Expansion

Special path variables are automatically expanded upon initialization:
- `${DATA_DIR}`: Resolves to the absolute path of `database.data_dir`.
- `${HOME}`: Resolves to the user's home directory (`os.environ["HOME"]`).

---

## Production Tuning Recommendations

1. **SQLite WAL Concurrency**:
   Keep `wal_mode: true` and `busy_timeout_ms: 10000` to prevent `sqlite3.OperationalError: database is locked` under high multi-process concurrency.
2. **Unix Domain Sockets**:
   When running on the same host, use `socket_path` instead of TCP for FalkorDB to reduce network stack overhead and latency by ~20-30%.
3. **Worker Semaphore**:
   Set `pipeline.concurrency_limit` according to your LLM API rate limits (e.g., 4 for local LiteLLM / Ollama, 8 for cloud Tier-3/4 APIs).
