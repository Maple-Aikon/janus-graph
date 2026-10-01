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
from typing import Any, Dict, List, Optional

from graphiti_core.helpers import normalize_l2
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_MMR
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
)
from graphiti_core.search.search import search as graphiti_search
from graphiti_core.search.search_utils import get_embeddings_for_edges

from ..config import (
    JanusSettings,
    resolve_cosine_gate_min,
    resolve_cosine_gate_overfetch,
    resolve_reranker_min_score,
    resolve_search_params,
)

logger = logging.getLogger("janus_graph.engine.search_memory")


async def _apply_cosine_gate(
    edges: List[Any],
    query: str,
    gate_min: float,
    limit: int,
    graphiti: Any,
    query_vector: Optional[List[float]],
) -> List[Any]:
    """Drop ``edges`` whose TRUE cosine similarity to the query is < ``gate_min``.

    Why this recomputes cosine instead of reading a score off the result
    object (2026-10-01 — three separate dead ends measured, not guessed):

      - ``SearchResults.edge_reranker_scores`` holds MMR scores, and MMR is
        ``lambda * cos + (lambda - 1) * max_sim`` — a relevance and
        redundancy MIXTURE. On-topic and off-topic MMR distributions
        overlap on this host (on-topic min 0.3191 < off-topic max 0.3281),
        so MMR can never act as a relevance gate.
      - The FalkorDB driver's own similarity score is ``(1 + cos) / 2``,
        not cosine. Comparing a cosine floor against it silently halves
        the effective threshold (floor F means cos >= 2F - 1).
      - The bm25/fulltext arm receives no ``min_score`` at all in
        graphiti_core, so lexical-only hits carry no comparable number.

    So the only sound signal is the raw embedding dot product, which is
    what ``get_embeddings_for_edges`` already loads for the MMR reranker —
    same source, same cypher, no new query needed.

    ``query_vector`` is passed in so the caller can embed the query once and
    hand the identical vector to both ``graphiti_search`` and this gate.
    That matters: graphiti_core's internal embed normalises the query with
    ``query.replace(chr(10), ' ')`` before embedding, so a gate that
    embedded the raw query would compute cosine against a *different*
    vector than the one search used. Pass-through mode skips that
    normalisation, so the caller is responsible for replicating it (see
    ``search_memory``).

    Edges with a missing/empty embedding cannot be scored. Rather than
    silently keeping them (which would defeat the gate for exactly the
    garbage this is meant to filter) or silently dropping them (which can
    hide legitimate results), they are dropped and counted in a log line,
    so a spike is visible instead of mysterious.

    Returns the filtered list, truncated back to ``limit``.
    """
    if not edges:
        return []

    if query_vector is None:
        query_vector = await graphiti.clients.embedder.create(
            input_data=[query.replace(chr(10), " ")]
        )
    if not query_vector:
        logger.warning(
            "cosine gate: empty query vector; returning %d edges ungated", len(edges)
        )
        return edges[:limit]

    try:
        embeddings = await get_embeddings_for_edges(graphiti.clients.driver, edges)
    except Exception as e:  # noqa: BLE001 — gate must never break search
        logger.warning(
            "cosine gate: could not load edge embeddings (%s: %s); "
            "returning %d edges ungated",
            type(e).__name__, e, len(edges),
        )
        return edges[:limit]

    q = normalize_l2(query_vector)
    kept: List[Any] = []
    unscorable = 0
    for edge in edges:
        vec = embeddings.get(getattr(edge, "uuid", None))
        if vec is None or len(vec) == 0:
            unscorable += 1
            continue
        try:
            cos = float(q @ normalize_l2(vec))
        except (TypeError, ValueError):
            unscorable += 1
            continue
        if cos >= gate_min:
            kept.append(edge)

    if unscorable:
        logger.info(
            "cosine gate: dropped %d/%d edges with missing fact_embedding",
            unscorable, len(edges),
        )
    logger.info(
        "cosine gate: floor=%.4f kept=%d/%d edges",
        gate_min, len(kept), len(edges),
    )
    return kept[:limit]


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
        # don't bleed state across concurrent calls. ``limit`` may be raised
        # below (cosine gate over-fetch); the caller's limit is restored when
        # truncating the gated result.
        search_config = copy.deepcopy(EDGE_HYBRID_SEARCH_MMR)

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

        # v0.4.8: push the reranker floor onto the SearchConfig itself.
        # It lives on the top-level config (not the per-entity sub-configs
        # above) and search.py:185-222 forwards it verbatim into the
        # reranker. Leaving it at the graphiti_core default of 0 makes the
        # MMR edge reranker filter out essentially every candidate, because
        # MMR scores are signed and land below 0 whenever mmr_lambda < 1.
        # This is the single line that decides whether MMR returns
        # anything at all — see SearchConfig.reranker_min_score.
        search_config.reranker_min_score = resolve_reranker_min_score(cfg)

        search_filter = SearchFilters(
            invalid_at=[[DateFilter(comparison_operator=ComparisonOperator.is_null)]],
        )

        # ── Post-hoc true-cosine gate (2026-10-01) ────────────────────
        # ``None`` keeps the pre-gate behaviour byte-for-byte: no extra
        # embedding call, no extra cypher, same edges, same count. Only
        # when the operator explicitly opts in do we pay for over-fetching.
        gate_min = resolve_cosine_gate_min(cfg)
        fetch_limit = limit
        query_vector: Optional[List[float]] = None
        if gate_min is not None:
            fetch_limit = max(1, limit * resolve_cosine_gate_overfetch(cfg))
            # Embed ONCE, here, and hand the same vector to both graphiti
            # and the gate. Must replicate graphiti_core's own newline
            # normalisation, because passing ``query_vector`` explicitly
            # BYPASSES the embed block that normally does it
            # (graphiti_core/search/search.py) — so without this the gate
            # would score a multi-line query against a different vector
            # than the one search itself used.
            query_vector = await graphiti.clients.embedder.create(
                input_data=[query.replace(chr(10), " ")]
            )

        search_config.limit = fetch_limit

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
            query_vector=query_vector,
        )

        edges = getattr(search_results, "edges", search_results) or []

        if gate_min is not None:
            edges = await _apply_cosine_gate(
                edges,
                query=query,
                gate_min=gate_min,
                limit=limit,
                graphiti=graphiti,
                query_vector=query_vector,
            )

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
