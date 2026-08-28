"""FastMCP Server providing standard memory tools."""

from __future__ import annotations

from typing import Any, Optional
from mcp.server.fastmcp import FastMCP
from ..config import JanusSettings, load_config
from ..pipeline.queue import EpisodeQueue


def create_mcp_server(settings: Optional[JanusSettings] = None) -> FastMCP:
    """Create FastMCP server instance with memory endpoints."""
    cfg = settings or load_config()
    mcp = FastMCP("janus-graph")
    queue = EpisodeQueue(cfg.pipeline.queue_db_path)

    @mcp.tool()
    async def add_episode(content: str, name: Optional[str] = None, source_description: Optional[str] = None, group_id: Optional[str] = None) -> str:
        """Enqueue an episode for asynchronous ingestion into Janus Knowledge Graph."""
        ep_id = queue.enqueue(content=content, group_id=group_id)
        return f"Episode enqueued: {ep_id}"

    @mcp.tool()
    async def queue_status() -> dict[str, Any]:
        """Check status of asynchronous queue and DLQ."""
        return queue.get_stats()

    @mcp.tool()
    async def health() -> dict[str, Any]:
        """Return system health status."""
        return {
            "status": "ok",
            "version": "0.1.0",
            "queue": queue.get_stats(),
        }

    return mcp
