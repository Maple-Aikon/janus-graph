"""MCP Server providing standard and extended Janus Knowledge Graph tools."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer

from ..config import JanusSettings, load_config
from ..pipeline.queue import EpisodeQueue
from ..pipeline.dream import run_dream_consolidation
from ..cache.embed_cache import EmbedCache

# graphiti-core: import SearchConfig recipe + SearchFilters/DateFilter for
# the MCP `search_memory` tool (v0.4.5: parity with hook Falkor path).
#   EDGE_HYBRID_SEARCH_MMR  = bm25 + cosine fused, then MMR rerank (lambda=0.5).
#   SearchFilters(invalid_at=IS NULL) filters out soft-deleted facts.
# Both are wired into graphiti.search() below; graphiti-core cascades the
# filters into the cypher WHERE clause (search_filters.py:180-209).
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_MMR
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters

logger = logging.getLogger("janus_graph.mcp")

BLOCKED_CYPHER_KEYWORDS = [
    "CREATE", "DELETE", "SET", "DROP", "REMOVE",
    "MERGE", "DETACH", "FOREACH", "CALL", "LOAD CSV"
]


def is_read_only_cypher(query: str) -> tuple[bool, Optional[str]]:
    """Check if a Cypher query is read-only."""
    for kw in BLOCKED_CYPHER_KEYWORDS:
        if re.search(rf"\b{kw}\b", query, re.IGNORECASE):
            return False, kw
    return True, None


def create_mcp_server(settings: Optional[JanusSettings] = None) -> MCPServer:
    """Create MCPServer instance with memory tools."""
    cfg = settings or load_config()
    mcp = MCPServer("janus-graph")
    queue = EpisodeQueue(cfg.pipeline.queue_db_path)

    # Lazy graphiti instance
    _graphiti_instance: Optional[Any] = None

    def get_graphiti():
        nonlocal _graphiti_instance
        if _graphiti_instance is None:
            from ..core.instance import create_graphiti_instance
            _graphiti_instance = create_graphiti_instance(cfg)
        return _graphiti_instance

    def get_falkordb_client():
        from falkordb import FalkorDB
        return FalkorDB(host=cfg.engine.host, port=cfg.engine.port)

    @mcp.tool()
    async def add_episode(
        content: str,
        name: Optional[str] = None,
        source_description: Optional[str] = "user_conversation",
    ) -> Dict[str, Any]:
        """Enqueue a new text episode into the Janus Knowledge Graph queue.

        Note: ``group_id`` is locked at the canonical ``cfg.graphiti.group_id``.
        It is not exposed as a tool parameter because PicoClaw's MCP tools must
        share a single memory tenant (graphiti_memory). Edit ``config.yaml`` /
        ``JANUS_GRAPHITI__GROUP_ID`` env to change.
        """
        target_group = cfg.graphiti.group_id
        ep_name = name or f"Episode {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"

        ep_id = await queue.enqueue(
            content=content,
            name=ep_name,
            source_description=source_description or "user_conversation",
            group_id=target_group,
        )
        return {
            "success": True,
            "queued": True,
            "episode_id": ep_id,
            "group_id": target_group,
            "message": "Episode accepted and queued. Processing handled asynchronously by worker cron.",
        }

    @mcp.tool()
    async def queue_status(episode_id: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        """Inspect the episode queue status, specific record, or dead-letter rows."""
        stats = queue.get_stats()
        out: Dict[str, Any] = {
            "stats": stats,
            "max_attempts": cfg.pipeline.max_attempts,
        }
        if episode_id:
            rec = queue.get_record(episode_id)
            if rec is None:
                out["episode"] = None
            else:
                out["episode"] = {
                    "id": rec.id,
                    "status": rec.status,
                    "attempt_count": rec.attempt_count,
                    "last_error": rec.last_error,
                    "checkpoint": rec.checkpoint,
                    "created_at": rec.created_at,
                    "updated_at": rec.updated_at,
                }
        else:
            dlq_records = queue.get_dlq_records(limit=limit)
            out["dlq_recent"] = [
                {
                    "id": r.get("episode_id"),
                    "status": "dead_letter",
                    "attempt_count": r.get("attempt_count"),
                    "last_error": r.get("last_error"),
                    "failed_at": r.get("failed_at"),
                }
                for r in dlq_records
            ]
        return out

    @mcp.tool()
    async def retry_dlq_episode(episode_id: str) -> Dict[str, Any]:
        """Manually replay a dead-letter episode back to queued status."""
        success = await queue.replay_dlq_episode(episode_id)
        if not success:
            return {
                "success": False,
                "episode_id": episode_id,
                "message": "Episode not in DLQ or not found.",
            }
        return {
            "success": True,
            "episode_id": episode_id,
            "message": "Episode re-queued successfully for processing by worker cron.",
        }

    @mcp.tool()
    async def graphiti_health() -> Dict[str, Any]:
        """Composite health snapshot: queue stats, DLQ counts, and engine reachability."""
        stats = queue.get_stats()
        engine_ok = False
        try:
            fdb = get_falkordb_client()
            fdb.connection.ping()
            engine_ok = True
        except Exception:
            engine_ok = False

        status = "ok" if (engine_ok and stats.get("dlq", 0) == 0) else "degraded"
        return {
            "status": status,
            "version": "0.1.0",
            "engine_reachable": engine_ok,
            "queue_stats": stats,
        }

    @mcp.tool()
    async def search_memory(
        query: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Search the knowledge graph using hybrid semantic retrieval (MMR reranked).

        v0.4.5: pipes SearchConfig=EDGE_HYBRID_SEARCH_MMR (BM25+cosine fused
        then MMR reranked) and SearchFilters(invalid_at=IS NULL) into
        graphiti.search(). This gives the MCP path parity with the hook
        Falkor path so recall quality is consistent regardless of which path
        wins the budget race.

        v0.4.6: ``sim_min_score`` and ``mmr_lambda`` are read from
        ``cfg.search`` (env ``JANUS_SEARCH__SIM_MIN_SCORE`` /
        ``JANUS_SEARCH__MMR_LAMBDA`` → config.yaml ``search.*`` → built-in
        defaults 0.6 / 0.5). The values are applied to every per-entity
        search config (edge / node / episode / community) so MMR rerank
        stays consistent across the parallel fan-out.

        Note: ``group_id`` is locked to ``cfg.graphiti.group_id`` to keep
        PicoClaw's MCP tools in a single memory tenant.
        """
        target_group = cfg.graphiti.group_id
        try:
            graphiti = get_graphiti()
            # SearchConfig recipe is a module-level singleton; mutate locally
            # so we don't bleed state across concurrent tool calls.
            import copy
            from janus_graph.config import resolve_search_params
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
            results = await graphiti.search(
                query=query,
                group_ids=[target_group],
                num_results=limit,
                search_config=search_config,
                search_filter=search_filter,
            )
            facts = []
            for edge in getattr(results, "edges", results) if isinstance(results, (list, tuple)) else getattr(results, "edges", []):
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
        except Exception as e:
            logger.error(f"search_memory error: {e}")
            return {
                "success": False,
                "error": str(e),
                "group_id": target_group,
                "query": query,
            }

    @mcp.tool()
    async def get_entity(entity_name: str) -> Dict[str, Any]:
        """Retrieve an entity and its direct relationships.

        Note: ``group_id`` is locked at the canonical ``cfg.graphiti.group_id``.
        Not exposed as a parameter to prevent cross-tenant memory mixing.
        """
        target_group = cfg.graphiti.group_id
        try:
            fdb = get_falkordb_client()
            g = fdb.select_graph(target_group)
            q = """
            MATCH (e:Entity) 
            WHERE toLower(e.name) CONTAINS toLower($name) 
            OPTIONAL MATCH (e)-[r]-(other:Entity)
            RETURN e.name, labels(e), collect(DISTINCT {rel: type(r), other: other.name}) LIMIT 50
            """
            res = g.query(q, {"name": entity_name})
            rows = []
            for row in res.result_set:
                rows.append({
                    "name": row[0],
                    "labels": row[1],
                    "relationships": row[2],
                })
            return {
                "success": True,
                "entity_name": entity_name,
                "group_id": target_group,
                "results": rows,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "entity_name": entity_name,
            }

    @mcp.tool()
    async def list_entities(limit: int = 50, filter_query: Optional[str] = None) -> Dict[str, Any]:
        """List entities stored in the knowledge graph with relationship counts.

        Note: ``group_id`` is locked at the canonical ``cfg.graphiti.group_id``.
        Not exposed as a parameter to prevent cross-tenant memory mixing.
        """
        target_group = cfg.graphiti.group_id
        try:
            fdb = get_falkordb_client()
            g = fdb.select_graph(target_group)
            if filter_query:
                q = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($filter)
                OPTIONAL MATCH (e)-[r]-()
                RETURN e.name, count(r) AS degree
                ORDER BY degree DESC LIMIT $limit
                """
                params = {"filter": filter_query, "limit": limit}
            else:
                q = """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r]-()
                RETURN e.name, count(r) AS degree
                ORDER BY degree DESC LIMIT $limit
                """
                params = {"limit": limit}
            res = g.query(q, params)
            entities = [{"name": r[0], "degree": r[1]} for r in res.result_set]
            return {
                "success": True,
                "group_id": target_group,
                "count": len(entities),
                "entities": entities,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "group_id": target_group,
            }

    @mcp.tool()
    async def cypher_query(query: str) -> Dict[str, Any]:
        """Execute a read-only openCypher query directly against FalkorDB.

        Note: ``group_id`` is locked at the canonical ``cfg.graphiti.group_id``.
        Not exposed as a parameter to prevent cross-tenant memory mixing.
        """
        is_ro, blocked_kw = is_read_only_cypher(query)
        if not is_ro:
            return {
                "success": False,
                "error": f"Mutating Cypher operations are forbidden in read-only mode (found '{blocked_kw}')",
                "blocked_keyword": blocked_kw,
            }

        target_group = cfg.graphiti.group_id
        try:
            fdb = get_falkordb_client()
            g = fdb.select_graph(target_group)
            res = g.query(query)
            headers = [h[1] for h in res.header] if res.header else []
            rows = res.result_set
            return {
                "success": True,
                "group_id": target_group,
                "headers": headers,
                "row_count": len(rows),
                "rows": rows,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "group_id": target_group,
            }

    @mcp.tool()
    async def run_dream_maintenance(force: bool = False) -> Dict[str, Any]:
        """Execute Graphiti Dream Mode memory consolidation cycle (Phases 1-4).

        Note: ``group_id`` is locked at the canonical ``cfg.graphiti.group_id``.
        Not exposed as a parameter to prevent cross-tenant memory mixing.
        """
        res = await run_dream_consolidation(cfg, force=force)
        return {
            "success": True,
            "report": res,
        }

    @mcp.tool()
    async def cache_stats() -> Dict[str, Any]:
        """Return stats of the embedding cache."""
        return {
            "enabled": cfg.cache.embed.enabled,
            "max_size": cfg.cache.embed.max_size,
            "eviction": cfg.cache.embed.eviction,
        }

    @mcp.tool()
    async def report_stats(limit: int = 20, kind: Optional[str] = None) -> Dict[str, Any]:
        """Read the most recent execution reports."""
        log_path = Path(cfg.report.sinks.file.path)
        if not log_path.exists():
            return {"reports": [], "total": 0}
        
        reports = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if kind and item.get("kind") != kind:
                        continue
                    reports.append(item)
                except json.JSONDecodeError:
                    continue
        
        reports = reports[-limit:]
        return {
            "reports": reports,
            "count": len(reports),
        }

    return mcp
