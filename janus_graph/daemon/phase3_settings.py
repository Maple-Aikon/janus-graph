"""Phase 3 config extensions for janus-graph daemon.

Adds two settings namespaces under ``cfg.daemon``:

  DaemonLockSettings     - 3-mode advisory-lock policy for /episodes ingest
                           (decides whether daemon POST /episodes may take
                           ownership of rows that MCP stdio also enqueues).
  DaemonSearchGraphSettings
                         - Phase 3.1 /search/graph endpoint knobs
                           (hard caps, timeouts, MMR defaults).

Per existing Plan #2 section 11 env precedence: ``JANUS_DAEMON__LOCK__*``
and ``JANUS_DAEMON__SEARCH_GRAPH__*`` override YAML keys, then class defaults.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


LockMode = Literal["mcp_only", "daemon_only", "dual_with_lock"]


class DaemonLockSettings(BaseModel):
    """3-mode advisory-lock policy for POST /episodes ingest.

    Modes:
      mcp_only        - daemon /episodes rejects (HTTP 409 LOCK_MODE_MCP_ONLY).
                         Existing MCP stdio path is sole writer.
      daemon_only     - daemon /episodes is sole writer; MCP enqueue must
                         route through HTTP. Rows ingested via daemon carry
                         claimed_by='daemon:<pid>'.
      dual_with_lock  - both writers OK. Row claimed_by set on insert; the
                         OTHER writer sees claimed_by != NULL and skips
                         (treated as duplicate, HTTP 409 LOCKED).

    Defaults to ``mcp_only`` so the Phase 2 PR is NOT a behavior change
    when Phase 3 lands - operators must explicitly opt-in via env / YAML
    before daemon starts accepting POST /episodes writes.
    """

    mode: LockMode = "mcp_only"
    instance_id_prefix: str = "daemon"  # claimed_by = "{prefix}:{pid}"


class DaemonSearchGraphSettings(BaseModel):
    """Phase 3.1 /search/graph endpoint configuration.

    Per Plan #2 section Phase 3.1. All knobs are POST-deploy
    tunable via ``JANUS_DAEMON__SEARCH_GRAPH__*`` env vars.
    """

    enabled: bool = True
    max_facts_per_query: int = 5000        # hard cap before rerank
    traversal_timeout_sec: float = 2.0    # 504 budget
    embedding_concurrency: int = 4        # semaphore for nomic-embed batch
    default_min_cosine: float = 0.5       # post-filter threshold
    default_mmr_lambda: float = 0.7       # MMR dominance (Plan section Phase 3.1)
    default_limit: int = 10               # response default
    max_limit: int = 50                   # hard ceiling on request limit
    warn_on_hop_cap: bool = True
    # Embedding service URL (OpenAI-compatible /embeddings endpoint).
    # Defaults match config.yaml embedding.base_url.
    embedding_base_url: str = "http://127.0.0.1:8081/v1"
    embedding_model: str = "nomic-embed-text-v2"
    embedding_api_key: str = "not-needed"
