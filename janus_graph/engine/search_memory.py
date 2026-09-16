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
    - Failure: any exception is caught and returned as ``{"success": False,
              "error": str(e), "group_id": ..., "query": ...}``. This
              preserves the existing MCP contract (caller logs but never
              raises to the MCP client).
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
        logger.error("search_memory error: %s", e)
        return {
            "success": False,
            "error": str(e),
            "group_id": target_group,
            "query": query,
        }
