# Janus-Graph 🏛️

**Janus-Graph** is a modular, high-resilience Knowledge Graph Memory Engine for autonomous AI agents, powered by FalkorDB, Graphiti Core, and SQLite WAL resilient pipelines.

## Key Features

- ⚡ **Embedded Knowledge Graph**: High-performance Graph storage & Cypher query engine via FalkorDB.
- 🛡️ **Schema-Repairing Heuristics**: Automatic self-repair for LLM output edge/entity validation discrepancies without re-querying models.
- 🔄 **Resilient Asynchronous Ingestion**: SQLite WAL-backed FIFO queue with exponential backoff, dead-letter queue (DLQ), and automated replay.
- 🌙 **Dream Mode Consolidation**: Autonomous memory clustering, 2-tier deduplication, orphan node pruning, and DLQ auto-repair.
- 🔌 **Standard Model Context Protocol (MCP)**: Seamless integration into agent ecosystems via FastMCP/stdio.
- 📊 **Multi-Channel Health Reporting**: Granular status reporting with file, CLI, Telegram, and Webhook sinks.

---

## Documentation

- 🏛️ **[System Architecture](docs/architecture.md)**: Layered architecture, subsystem breakdown, and ingestion/query lifecycles.
- ⚙️ **[Configuration Guide](docs/configuration.md)**: Full reference for `janus_graph.yaml`, `.env` variables, and production tuning.
- 🛡️ **[Schema Heuristics Guide](docs/heuristics.md)**: Pydantic validation interceptor, built-in rules, and authoring custom repairs.
- 🛠️ **[Operations & Runbook](docs/operations.md)**: CLI usage, background cron jobs, online snapshots, and disaster rollback.
- 🔌 **[MCP Protocol & Integration](docs/mcp-protocol.md)**: FastMCP tools reference and client setup (PicoClaw, Claude Desktop).

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/maple-aikon/janus-graph.git
cd janus-graph

# Setup virtual environment and dependencies using uv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,cli]"

# Download or verify pre-built engine binaries
./bin/install.sh
```

### Configuration

Copy the example configuration file:

```bash
cp config.example.yaml config.yaml
```

### CLI Usage

```bash
# Run system and environment health checks
janus-graph doctor

# Start embedded FalkorDB engine
janus-graph engine start

# Process queued episodes
janus-graph sweep

# Run memory dream consolidation cycle
janus-graph dream

# Start MCP Server
janus-graph mcp
```

---

## Binary Distribution & Licenses

- `janus-graph` core codebase is licensed under the **Apache License 2.0**.
- Pre-built `falkordb.so` binary is derived from official FalkorDB releases (BSD-3-Clause). See `bin/LICENSE-FALKORDB`.
- Pre-built `redis-server-8` binary is licensed under the Redis Source Available License v2 (RSALv2). See `bin/LICENSE-REDIS`.
