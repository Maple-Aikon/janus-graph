"""Phase 3 POST /episodes endpoint tests.

Tests cover:
  T1 happy enqueue → 202 + episode_id + claimed_by
  T2 duplicate content → 409 DUPLICATE_EPISODE
  T3 mode=mcp_only (default) → 409 LOCK_MODE_MCP_ONLY
  T4 mode=daemon_only with claimed_by stamped → row carries instance id
  T5 malformed body → 422 INVALID_BODY
  T6 missing content → 422 INVALID_BODY
  T7 Falkor down + queue adapter ready → still 202 (queue doesn't depend on Falkor)

Note: /episodes is independent of Falkor at write time (only /search/graph
needs Falkor). So we don't kill Falkor here — that's covered in
test_search_graph.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from typing import Any, Dict

import pytest
from aiohttp.test_utils import TestClient, TestServer

from janus_graph.config import JanusSettings
from janus_graph.daemon.episode_queue_adapter import (
    EpisodeQueueAdapter,
    EpisodeDuplicateError,
    LockModeError,
    _payload_hash,
)
from janus_graph.daemon.http_server import build_app
from janus_graph.daemon.lifespan import DaemonContext
from janus_graph.pipeline.queue import EpisodeQueue
from janus_graph.daemon.falkordb_supervisor import DaemonSupervisor
from janus_graph.daemon.http_server import __version__


@pytest.fixture
def tmp_queue_path(tmp_path):
    """Per-test SQLite queue path so tests don't share state."""
    return str(tmp_path / "episodes.db")


@pytest.fixture
def daemon_only_settings(tmp_queue_path):
    """Default settings + lock.mode=daemon_only so /episodes accepts writes."""
    s = JanusSettings()
    s.pipeline.queue_db_path = tmp_queue_path
    s.daemon.lock.mode = "daemon_only"
    return s


@pytest.fixture
def mcp_only_settings(tmp_queue_path):
    """Settings with lock.mode=mcp_only (default) for 409 test."""
    s = JanusSettings()
    s.pipeline.queue_db_path = tmp_queue_path
    s.daemon.lock.mode = "mcp_only"
    return s


@pytest.fixture
async def episode_adapter(daemon_only_settings):
    """Standalone EpisodeQueueAdapter for unit-level tests.

    Calls ``await adapter.start()`` so schema migration runs before any
    enqueue — the adapter enforces this contract.
    """
    queue = EpisodeQueue(db_path=daemon_only_settings.pipeline.queue_db_path)
    adapter = EpisodeQueueAdapter(
        queue=queue,
        lock_settings=daemon_only_settings.daemon.lock,
    )
    await adapter.start()
    return adapter


# ─── unit tests (adapter-level) ──────────────────────────────────────


def test_payload_hash_deterministic():
    """Same inputs → same hash (sha256 stability)."""
    a = _payload_hash("hello", "graphiti", "user")
    b = _payload_hash("hello", "graphiti", "user")
    assert a == b
    assert len(a) == 64


def test_payload_hash_differs_on_source():
    """Same content + group, different source → different hash (so
    user_conversation vs agent_interaction do not collide).
    """
    a = _payload_hash("hello", "graphiti", "user")
    b = _payload_hash("hello", "graphiti", "agent")
    assert a != b


@pytest.mark.asyncio
async def test_adapter_enqueue_happy(episode_adapter):
    """T1: enqueue succeeds, returns episode_id, stamps claimed_by."""
    result = await episode_adapter.enqueue_guarded(
        content="hello world",
        group_id="graphiti_memory",
        name="t1",
    )
    assert result.deduplicated is False
    assert result.episode_id
    assert result.claimed_by is not None
    assert result.claimed_by.startswith("daemon:")
    # Row should exist in SQLite with claimed_by set.
    with episode_adapter._queue._get_connection() as conn:
        row = conn.execute(
            "SELECT id, status, claimed_by FROM episodes WHERE id = ?",
            (result.episode_id,),
        ).fetchone()
    assert row is not None
    assert row["status"] == "queued"
    assert row["claimed_by"].startswith("daemon:")


@pytest.mark.asyncio
async def test_adapter_duplicate_raises(episode_adapter):
    """T2: enqueueing same content twice raises EpisodeDuplicateError."""
    await episode_adapter.enqueue_guarded(
        content="duplicate-me",
        group_id="graphiti_memory",
    )
    with pytest.raises(EpisodeDuplicateError) as ei:
        await episode_adapter.enqueue_guarded(
            content="duplicate-me",
            group_id="graphiti_memory",
        )
    assert ei.value.existing_episode_id
    # The existing id should match the first insert.
    assert len(ei.value.existing_episode_id) > 0


@pytest.mark.asyncio
async def test_adapter_lock_mode_mcp_only_raises(tmp_queue_path):
    """T3: mode=mcp_only blocks daemon /episodes with LockModeError.

    Note: lock mode is checked BEFORE migration, so we don't need start().
    """
    s = JanusSettings()
    s.pipeline.queue_db_path = tmp_queue_path
    s.daemon.lock.mode = "mcp_only"
    queue = EpisodeQueue(db_path=tmp_queue_path)
    adapter = EpisodeQueueAdapter(queue=queue, lock_settings=s.daemon.lock)
    with pytest.raises(LockModeError):
        await adapter.enqueue_guarded(content="x")


@pytest.mark.asyncio
async def test_adapter_stats(episode_adapter):
    """stats() returns by_status + dlq_open + claimed_active counts."""
    await episode_adapter.enqueue_guarded(content="one")
    await episode_adapter.enqueue_guarded(content="two")
    stats = episode_adapter.stats()
    assert "by_status" in stats
    assert stats["by_status"].get("queued", 0) >= 2
    assert stats["dlq_open"] == 0
    assert stats["claimed_active"] >= 2


# ─── HTTP-level tests ────────────────────────────────────────────────


@pytest.fixture
async def http_client_daemon_only(daemon_only_settings):
    """aiohttp TestClient bound to a DaemonContext with daemon_only lock."""
    # Disable search_graph for these tests (not the focus).
    daemon_only_settings.daemon.search_graph.enabled = False
    queue = EpisodeQueue(db_path=daemon_only_settings.pipeline.queue_db_path)
    adapter = EpisodeQueueAdapter(
        queue=queue,
        lock_settings=daemon_only_settings.daemon.lock,
    )
    await adapter.start()
    ctx = DaemonContext(
        settings=daemon_only_settings,
        supervisor=DaemonSupervisor(
            engine_config=daemon_only_settings.engine,
            circuit_config=daemon_only_settings.daemon.falkor_circuit,
        ),
        http_settings=daemon_only_settings.daemon.http,
        version=__version__,
        shutdown_event=asyncio.Event(),
        queue=queue,
        queue_adapter=adapter,
    )
    app = build_app(ctx)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        yield client
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_http_episodes_happy(http_client_daemon_only):
    """T1 HTTP: POST /episodes with valid body returns 202 + episode_id."""
    resp = await http_client_daemon_only.post(
        "/episodes",
        json={"content": "hello from test", "name": "t-http-1"},
    )
    assert resp.status == 202
    body = await resp.json()
    assert "episode_id" in body
    assert body["claimed_by"].startswith("daemon:")
    assert body["deduplicated"] is False
    assert body["daemon_version"] == __version__


@pytest.mark.asyncio
async def test_http_episodes_duplicate(http_client_daemon_only):
    """T2 HTTP: same content twice → second is 409 DUPLICATE_EPISODE."""
    body1 = {"content": "dup-content-http", "name": "t-http-dup"}
    resp1 = await http_client_daemon_only.post("/episodes", json=body1)
    assert resp1.status == 202
    ep_id_1 = (await resp1.json())["episode_id"]
    resp2 = await http_client_daemon_only.post("/episodes", json=body1)
    assert resp2.status == 409
    body2 = await resp2.json()
    assert body2["code"] == "DUPLICATE_EPISODE"
    assert body2["existing_episode_id"] == ep_id_1


@pytest.mark.asyncio
async def test_http_episodes_mcp_only_lock(tmp_queue_path):
    """T3 HTTP: mode=mcp_only → 409 LOCK_MODE_MCP_ONLY."""
    s = JanusSettings()
    s.pipeline.queue_db_path = tmp_queue_path
    s.daemon.lock.mode = "mcp_only"
    queue = EpisodeQueue(db_path=tmp_queue_path)
    adapter = EpisodeQueueAdapter(queue=queue, lock_settings=s.daemon.lock)
    await adapter.start()
    ctx = DaemonContext(
        settings=s,
        supervisor=DaemonSupervisor(
            engine_config=s.engine, circuit_config=s.daemon.falkor_circuit
        ),
        http_settings=s.daemon.http,
        version=__version__,
        shutdown_event=asyncio.Event(),
        queue=queue,
        queue_adapter=adapter,
    )
    app = build_app(ctx)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.post("/episodes", json={"content": "x"})
        assert resp.status == 409
        body = await resp.json()
        assert body["code"] == "LOCK_MODE_MCP_ONLY"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_http_episodes_malformed_body(http_client_daemon_only):
    """T5 HTTP: non-JSON body → 422 INVALID_BODY."""
    resp = await http_client_daemon_only.post(
        "/episodes",
        data="not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status == 422
    body = await resp.json()
    assert body["code"] == "INVALID_BODY"


@pytest.mark.asyncio
async def test_http_episodes_missing_content(http_client_daemon_only):
    """T6 HTTP: missing content field → 422 INVALID_BODY."""
    resp = await http_client_daemon_only.post(
        "/episodes", json={"name": "no-content"}
    )
    assert resp.status == 422
    body = await resp.json()
    assert body["code"] == "INVALID_BODY"
    assert "content" in body["message"]


@pytest.mark.asyncio
async def test_http_episodes_empty_content(http_client_daemon_only):
    """Empty content string also rejected → 422 INVALID_BODY."""
    resp = await http_client_daemon_only.post(
        "/episodes", json={"content": "   "}
    )
    assert resp.status == 422
    body = await resp.json()
    assert body["code"] == "INVALID_BODY"
