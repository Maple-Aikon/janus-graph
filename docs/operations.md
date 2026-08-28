# Janus-Graph Operations & Runbook 🛠️

This guide covers daily maintenance, daemon process supervision, cron pipeline sweeps, backup/recovery automation, and incident troubleshooting for **Janus-Graph**.

---

## CLI Cheatsheet

`janus-graph` provides a unified command-line interface (`janus_graph.cli.main`):

```bash
# 1. Environment & Infrastructure Doctor
janus-graph doctor

# 2. FalkorDB Engine Lifecycle Management
janus-graph engine start --daemon
janus-graph engine status
janus-graph engine stop

# 3. Asynchronous Queue Processing (Sweep)
janus-graph sweep --batch-size 20 --concurrency 4

# 4. Dream Mode Memory Consolidation
janus-graph dream --force

# 5. MCP Server (FastMCP stdio)
janus-graph mcp

# 6. Database Snapshot
janus-graph snapshot --target-dir ./backups

# 7. Database Rollback / Restore
janus-graph rollback ./backups/snapshot-20260828-120000
```

---

## Automated Operations Scripts

Production-ready bash scripts are located in `sources/janus-graph/bin/`:

### 1. `bin/snapshot-episodes-db.sh`
Performs an online, zero-downtime, transactionally consistent snapshot of the SQLite database and checkpoints directory.
- Uses `sqlite3 ... ".backup <target>"` (safe under concurrent WAL write transactions).
- Computes SHA256 checksums for backup integrity verification.
- Generates a structured `metadata.json` containing total episodes count, queue statuses, timestamp, and host architecture.

```bash
./bin/snapshot-episodes-db.sh /var/backups/janus-graph
```

### 2. `bin/rollback-janus-graph.sh`
Restores an existing snapshot cleanly.
- Stops running workers safely.
- Removes existing lock files (`.db-wal`, `.db-shm`).
- Restores `episodes.db` and associated checkpoints with checksum verification.

```bash
./bin/rollback-janus-graph.sh /var/backups/janus-graph/snapshot-20260828-120000
```

### 3. `bin/migrate-data.sh`
Handles migration of legacy Graphiti SQLite databases and checkpoints to new paths.

```bash
./bin/migrate-data.sh /legacy/path/episodes.db ./data/episodes.db
```

---

## Cron & Background Jobs

### 1. Ingestion Queue Sweep (Every 5-10 Minutes)
Configure cron to sweep pending episodes continuously:

```crontab
*/5 * * * * cd /path/to/janus-graph && .venv/bin/janus-graph sweep >> data/logs/cron_sweep.log 2>&1
```

### 2. Nightly Dream Mode Consolidation (02:30 AM Daily)
Executes Leiden graph community clustering, entity deduplication, orphan pruning, and DLQ auto-repair:

```crontab
30 2 * * * cd /path/to/janus-graph && .venv/bin/janus-graph dream >> data/logs/cron_dream.log 2>&1
```

---

## Health Checks & Incident Troubleshooting

### Diagnosing Issues via `janus-graph doctor`
Run `janus-graph doctor` to perform an automated health audit:
1. **Python Runtime**: Validates Python version (>=3.10) and packages.
2. **Binary Architecture Compatibility**: Ensures `redis-server-8` and `falkordb.so` match the CPU architecture (`aarch64` / `x86_64`).
3. **FalkorDB Daemon**: Checks TCP port 6379 and Unix socket connectivity.
4. **Database & WAL State**: Verifies SQLite integrity (`PRAGMA integrity_check`) and permissions.
5. **Embedding & LLM Endpoints**: Probes HTTP status and responsiveness.

### Common Incident Runbooks

#### Issue 1: `sqlite3.OperationalError: database is locked`
- **Cause**: Multi-process contention without WAL mode or insufficient timeout.
- **Fix**: Check that `wal_mode: true` is enabled in `janus_graph.yaml` and verify `busy_timeout_ms` >= 5000.

#### Issue 2: Episodes stuck in `processing` status
- **Cause**: Process died or timed out during Graphiti extraction.
- **Fix**: The reaper automatically runs during `janus-graph sweep` and resets expired leases (`> 600s`) back to `queued`.

#### Issue 3: Episodes accumulating in Dead-Letter Queue (`aborted` / `failed`)
- **Action 1**: Inspect the error cause via `janus_graph` CLI or MCP `queue_status(limit=20)`.
- **Action 2**: If the error was caused by a transient LLM schema issue, inspect `data/logs/llm_schema_quirks.jsonl`, implement a new heuristic rule in `janus_graph/heuristics/rules/`, and invoke MCP `retry_dlq_episode(episode_id)`.
