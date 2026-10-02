"""Unit tests for Janus-Graph MCPServer and tool handlers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from janus_graph.config import JanusSettings
from janus_graph.mcp.server import create_mcp_server, is_read_only_cypher
from janus_graph.pipeline.queue import EpisodeQueue


def test_is_read_only_cypher():
    # Safe queries
    assert is_read_only_cypher("MATCH (n) RETURN n")[0] is True
    assert (
        is_read_only_cypher("MATCH (e:Entity)-[r]->(o) RETURN e.name, type(r), o.name LIMIT 10")[0]
        is True
    )

    # Mutating queries
    assert is_read_only_cypher("CREATE (n:Test {name: 'foo'})")[0] is False
    assert is_read_only_cypher("MATCH (n) DELETE n")[0] is False
    assert is_read_only_cypher("MATCH (n) SET n.name = 'bar'")[0] is False
    assert is_read_only_cypher("MERGE (n:Entity {id: 1})")[0] is False
    assert is_read_only_cypher("MATCH (n) DETACH DELETE n")[0] is False


def _get_struct(res) -> dict:
    if getattr(res, "structured_content", None):
        return res.structured_content.get("result", res.structured_content)
    if getattr(res, "content", None) and len(res.content) > 0:
        return json.loads(res.content[0].text)
    return {}


@pytest.mark.asyncio
async def test_mcp_server_tools_registered(temp_dir: Path):
    db_path = temp_dir / "episodes.db"
    cfg = JanusSettings()
    cfg.pipeline.queue_db_path = str(db_path)

    mcp = create_mcp_server(cfg)
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]

    expected = [
        "add_episode",
        "queue_status",
        "retry_dlq_episode",
        "graphiti_health",
        "search_memory",
        "get_entity",
        "list_entities",
        "cypher_query",
        "run_dream_maintenance",
        "cache_stats",
        "report_stats",
    ]
    for exp in expected:
        assert exp in tool_names, f"Missing tool: {exp}"


@pytest.mark.asyncio
async def test_mcp_add_episode_and_status(temp_dir: Path):
    db_path = temp_dir / "episodes.db"
    cfg = JanusSettings()
    cfg.pipeline.queue_db_path = str(db_path)

    mcp = create_mcp_server(cfg)

    # Test add_episode
    res = await mcp.call_tool(
        "add_episode", {"content": "User likes matcha latte", "name": "Pref episode"}
    )
    assert res.is_error is False
    struct = _get_struct(res)
    assert struct["success"] is True
    assert struct["queued"] is True
    ep_id = struct["episode_id"]
    assert ep_id is not None

    # Test queue_status with episode_id
    status_res = await mcp.call_tool("queue_status", {"episode_id": ep_id})
    assert status_res.is_error is False
    status_struct = _get_struct(status_res)
    assert status_struct["episode"]["id"] == ep_id
    assert status_struct["episode"]["status"] == "queued"

    # Test queue_status overall
    overall = await mcp.call_tool("queue_status", {})
    overall_struct = _get_struct(overall)
    assert overall_struct["stats"]["queued"] == 1


@pytest.mark.asyncio
async def test_mcp_cache_and_report_stats(temp_dir: Path):
    log_path = temp_dir / "reports.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-08-28T21:00:00Z",
                "kind": "cron_sweep",
                "severity": "info",
                "summary": "All good in MCP test",
            }
        )
        + "\n"
    )

    cfg = JanusSettings()
    cfg.pipeline.queue_db_path = str(temp_dir / "episodes.db")
    cfg.report.sinks.file.path = str(log_path)
    mcp = create_mcp_server(cfg)

    # Test cache_stats
    c_res = await mcp.call_tool("cache_stats", {})
    c_struct = _get_struct(c_res)
    assert c_struct["enabled"] is True
    assert c_struct["max_size"] == 10000

    # Test report_stats
    r_res = await mcp.call_tool("report_stats", {"limit": 5})
    r_struct = _get_struct(r_res)
    assert r_struct["count"] == 1
    assert r_struct["reports"][0]["summary"] == "All good in MCP test"


@pytest.mark.asyncio
async def test_mcp_cypher_query_blocks_mutation(temp_dir: Path):
    cfg = JanusSettings()
    cfg.pipeline.queue_db_path = str(temp_dir / "episodes.db")
    mcp = create_mcp_server(cfg)

    res = await mcp.call_tool("cypher_query", {"query": "CREATE (n:Node) RETURN n"})
    struct = _get_struct(res)
    assert struct["success"] is False
    assert struct["blocked_keyword"] == "CREATE"


def test_bare_settings_never_touch_the_fallback_home(monkeypatch, tmp_path: Path):
    """Opening a queue on a default-constructed config must not write to ~/.janus-graph.

    ``_resolve_home()`` falls back to ``~/.janus-graph`` whenever
    JANUS_GRAPH_HOME is unset, so a bare ``JanusSettings()`` resolves
    ``queue_db_path`` to a real path outside the repo. Two tests in this file
    used to build a real EpisodeQueue from such a config, which recreated a
    0-row episodes.db on every pytest run — an empty database that reads
    exactly like a healthy idle queue. Hence the explicit pin at each call site.

    Note the split, verified 2026-10-02: ``JanusSettings()`` itself only
    mkdir's the parent (data/, data/falkordb/, data/logs/); the .db FILE is
    created by EpisodeQueue. So the directory is expected to appear here and
    only the file is the leak.
    """
    home = tmp_path / "fake_home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("JANUS_GRAPH_HOME", raising=False)

    cfg = JanusSettings()
    fb = home / ".janus-graph"
    assert cfg.pipeline.queue_db_path == str(fb / "data" / "episodes.db")

    # Constructing the settings makes the parent dir, but no database.
    assert fb.is_dir()
    assert not (fb / "data" / "episodes.db").exists()

    # Pinning is what keeps EpisodeQueue away from the fallback.
    pinned = tmp_path / "pinned.db"
    cfg.pipeline.queue_db_path = str(pinned)
    EpisodeQueue(db_path=cfg.pipeline.queue_db_path)
    assert pinned.exists(), "the pinned queue was not created"
    assert not (fb / "data" / "episodes.db").exists(), "test wrote into the fallback HOME"


def test_unpinned_queue_would_write_to_the_fallback_home(monkeypatch, tmp_path: Path):
    """Negative control: proves the assertion above can actually fail.

    Without this, the test above would also pass if the fallback simply
    stopped existing for some unrelated reason.
    """
    home = tmp_path / "fake_home2"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("JANUS_GRAPH_HOME", raising=False)

    cfg = JanusSettings()
    EpisodeQueue(db_path=cfg.pipeline.queue_db_path)  # deliberately NOT pinned

    leaked = home / ".janus-graph" / "data" / "episodes.db"
    assert leaked.exists(), (
        "the unpinned path no longer leaks — update the fix in "
        "test_mcp_cache_and_report_stats / test_mcp_cypher_query_blocks_mutation"
    )
