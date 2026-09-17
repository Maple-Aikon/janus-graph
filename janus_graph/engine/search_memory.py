"""Engine facade — single source of truth for knowledge-graph search_memory.

v0.5.0: extracted from ``janus_graph.mcp.server.search_memory`` so both
transports (stdio MCP and HTTP daemon) call the same code path. The MCP
tool becomes a thin adaptor; future PRs will wire the daemon HTTP
``/search/memory`` endpoint to this same facade (PR 2).

Contract:
    - Input:  cfg (JanusSettings), query (str), limit (int).
              Optional ``graphiti`` (caller passes its own singleton, e.g.
              the MCP closure-local one; otherwise lazy-init via
              ``core.instance.create_graphiti_instance``).
              Optional ``embedder`` is reserved for PR 2 — currently unused
              because module-level ``graphiti_search`` re-embeds via the
              graphiti instance's clients.
    - Output: dict with same wire shape as MCP tool
              (success/group_id/query/count/results) OR error dict on
              exception. The caller decides whether to log/return as-is.
    - Failure: any exception is caught and returned as
              ``{"success": False, "code", "error", "error_type",
              "group_id", "query"}``. The ``code`` field (v0.5.0.1 fix
              #4) is one of ``FALKOR_DISCONNECTED``, ``BACKEND_TIMEOUT``,
              ``INTERNAL_ERROR`` so callers can map to HTTP status without
              substring-matching the human-readable error string. The
              ``code`` field is ADDITIVE — existing MCP callers that only
              read ``success``/``error`` are unaffected.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional

from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_MMR
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
)
from graphiti_core.search.search import search as graphiti_search

from ..config import JanusSettings, resolve_search_params

logger = logging.getLogger("janus_graph.engine.search_memory")


async def search_memory(
    cfg: JanusSettings,
    query: str,
    limit: int,
    *,
    graphiti: Optional[Any] = None,
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Single SoT for MCP + HTTP search_memory.

    Returns: same JSON shape as current mcp/server.py:243-249
        {"success": True, "group_id", "query", "count",
         "results": [{"fact", "name", "valid_at", "invalid_at"}]}
    """
    target_group = cfg.graphiti.group_id
    try:
        # Lazy-init graphiti singleton if caller didn't pass one.
        # MCP path uses its own closure-local singleton; HTTP path will
        # pass the daemon's ctx.graphiti_instance.
        if graphiti is None:
            from ..core.instance import create_graphiti_instance

            graphiti = create_graphiti_instance(cfg)

        # SearchConfig recipe is a module-level singleton; deep-copy so we
        # don't bleed state across concurrent calls.
        search_config = copy.deepcopy(EDGE_HYBRID_SEARCH_MMR)
        search_config.limit = limit

        # v0.4.6: push cfg.search overrides (with fail-soft env validation)
        # down to every per-entity config.
        sim_min_score, mmr_lambda = resolve_search_params(cfg)
        for sub_cfg in (
            search_config.edge_config,
            search_config.node_config,
            search_config.episode_config,
            search_config.community_config,
        ):
            if sub_cfg is None:
                continue
            sub_cfg.sim_min_score = sim_min_score
            sub_cfg.mmr_lambda = mmr_lambda

        search_filter = SearchFilters(
            invalid_at=[[DateFilter(comparison_operator=ComparisonOperator.is_null)]],
        )

        # graphiti_core 0.20.x: Graphiti.search() does NOT accept
        # `search_config` kwarg (it hardcodes EDGE_HYBRID_SEARCH_RRF).
        # Use the module-level `search()` function which IS the function
        # that consumes SearchConfig (per graphiti_core.search.search).
        search_results = await graphiti_search(
            clients=graphiti.clients,
            query=query,
            group_ids=[target_group],
            config=search_config,
            search_filter=search_filter,
        )

        edges = getattr(search_results, "edges", search_results) or []
        facts: list[Dict[str, str]] = []
        for edge in edges:
            facts.append({
                "fact": getattr(edge, "fact", str(edge)),
                "name": getattr(edge, "name", ""),
                "valid_at": str(getattr(edge, "valid_at", "")),
                "invalid_at": str(getattr(edge, "invalid_at", "")),
            })

        return {
            "success": True,
            "group_id": target_group,
            "query": query,
            "count": len(facts),
            "results": facts,
        }
    except Exception as e:  # noqa: BLE001 — preserve MCP wire contract
        # v0.5.0.1 fix #4: classify exception so the HTTP handler can
        # map to status code WITHOUT substring-matching the message.
        # Old contract: caller string-matches "falkor"/"redis" etc. —
        # fragile, breaks on any future error message change.
        # New contract: emit ``code`` field with one of the typed
        # values below; handler prefers ``code`` then falls back to
        # substring for backward compat with pre-v0.5.0.1 callers.
        err_msg = str(e)
        err_type = type(e).__name__
        # Falkor disconnect / network errors: graphiti-core wraps
        # these as ``ConnectionError``, ``RedisConnectionError``,
        # ``OSError``, etc. The DNS-resolution / transport failure
        # path also lands here. Source-level: any graphiti-core error
        # whose message or type mentions falkor/redis/disconnect/
        # connection is treated as backend-down.
        lower_msg = err_msg.lower()
        if any(
            tok in lower_msg
            for tok in ("falkor", "redis", "disconnect", "connection refused",
                        "broken pipe", "reset by peer")
        ) or any(
            cls in err_type for cls in (
                "ConnectionError", "ConnectionRefusedError",
                "BrokenPipeError", "RedisConnectionError",
            )
        ):
            code = "FALKOR_DISCONNECTED"
        elif "timeout" in lower_msg or err_type == "TimeoutError":
            code = "BACKEND_TIMEOUT"
        else:
            code = "INTERNAL_ERROR"
        logger.error(
            "search_memory error: type=%s code=%s msg=%s",
            err_type, code, err_msg,
        )
        return {
            "success": False,
            "code": code,
            "error": err_msg,
            "error_type": err_type,
            "group_id": target_group,
            "query": query,
        }
