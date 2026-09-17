"""Tests for daemon/http_search_memory_handler.py (PR 2 / v0.5.0).

Covers:
  - Test A: query validation (length, type, range)
  - Test B: limit validation (int parsing, range)
  - Test C: rate limit F5 (60 req/min/IP token bucket)
  - Test D: 503 SEARCH_BACKEND_DOWN when ctx.graphiti_instance is None
  - Test E: 503 FALKOR_DISCONNECTED when supervisor.probe() returns False
  - Test F: 200 happy path — engine delegation produces correct shape
  - Test G: engine error envelope mapping (500 INTERNAL_ERROR vs
    503 FALKOR_DISCONNECTED based on error string)
  - Test H: response shape parity with stdio MCP ``search_memory`` tool
    (same keys: success, group_id, query, count, results; HTTP adds
    elapsed_ms + degraded).
  - Test I: F8 disconnect cancellation (aiohttp framework-level — verify
    no speculative cancellation code is added; mock engine and confirm
    no manual CancelledError try/except in handler).
  - Test J: rate limit Retry-After header present on 429.

These tests do NOT need a live Falkor — they mock ``ctx.supervisor.probe``
and ``engine_search_memory``. Plan §2.5 calls for an env-gated integration
test (Falkor + llama-server live); that's a separate file.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# pytest-aiohttp isn't installed; we drive the handler as a plain function
# with a mock request. The async parts are tested by ``await handler(req)``
# directly inside pytest-asyncio.
from aiohttp.test_utils import make_mocked_request

from janus_graph.config import JanusSettings
from janus_graph.daemon.http_search_memory_handler import (
    _RATE_LIMITER,
    _RATE_LIMIT_REQUESTS,
    _RATE_LIMIT_WINDOW_SEC,
    search_memory_handler,
)
from janus_graph.daemon.lifespan import DaemonContext


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def base_settings() -> JanusSettings:
    return JanusSettings()


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Each test starts with a clean rate limit bucket.

    Critical for Test C — module-level singleton state would otherwise
    bleed between tests.
    """
    _RATE_LIMITER.reset()
    yield
    _RATE_LIMITER.reset()


@pytest.fixture
def mock_graphiti() -> MagicMock:
    return MagicMock(name="graphiti")


@pytest.fixture
def alive_supervisor() -> MagicMock:
    """Mock supervisor whose probe() returns True (Falkor alive)."""
    sup = MagicMock()
    sup.probe = AsyncMock(return_value=True)
    return sup


@pytest.fixture
def dead_supervisor() -> MagicMock:
    """Mock supervisor whose probe() returns False (Falkor dead)."""
    sup = MagicMock()
    sup.probe = AsyncMock(return_value=False)
    return sup


def _make_ctx(
    base_settings: JanusSettings,
    supervisor: MagicMock,
    graphiti: Optional[Any],
) -> DaemonContext:
    """Build a DaemonContext with the fields the handler actually reads."""
    return DaemonContext(
        settings=base_settings,
        supervisor=supervisor,
        http_settings=base_settings.daemon.http,
        version="test-v0.5.0",
        shutdown_event=MagicMock(),
        graphiti_instance=graphiti,
    )


def _make_request(
    ctx: DaemonContext, query_string: str = ""
) -> Any:
    """Build a mocked aiohttp request bound to ctx + query string."""
    return make_mocked_request(
        "GET", f"/search/memory?{query_string}", app={"ctx": ctx}
    )


# ─── Test A: query validation ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_query_returns_400(
    base_settings, alive_supervisor, mock_graphiti
):
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)
    req = _make_request(ctx, "")  # no query
    resp = await search_memory_handler(req)
    assert resp.status == 400
    body = json.loads(resp.text)
    assert body["code"] == "INVALID_QUERY"


@pytest.mark.asyncio
async def test_short_query_returns_400(
    base_settings, alive_supervisor, mock_graphiti
):
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)
    req = _make_request(ctx, "query=hi")  # len 2 < 4
    resp = await search_memory_handler(req)
    assert resp.status == 400
    body = json.loads(resp.text)
    assert body["code"] == "INVALID_QUERY"
    assert "4" in body["message"] and "512" in body["message"]


@pytest.mark.asyncio
async def test_too_long_query_returns_400(
    base_settings, alive_supervisor, mock_graphiti
):
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)
    req = _make_request(ctx, f"query={'x' * 513}")
    resp = await search_memory_handler(req)
    assert resp.status == 400
    body = json.loads(resp.text)
    assert body["code"] == "INVALID_QUERY"


@pytest.mark.asyncio
async def test_query_whitespace_only_returns_400(
    base_settings, alive_supervisor, mock_graphiti
):
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)
    req = _make_request(ctx, "query=%20%20%20")  # 3 spaces
    resp = await search_memory_handler(req)
    assert resp.status == 400
    body = json.loads(resp.text)
    assert body["code"] == "INVALID_QUERY"


# ─── Test B: limit validation ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_limit_non_integer_returns_400(
    base_settings, alive_supervisor, mock_graphiti
):
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)
    req = _make_request(ctx, "query=Thuy%20Vi&limit=abc")
    resp = await search_memory_handler(req)
    assert resp.status == 400
    body = json.loads(resp.text)
    assert body["code"] == "INVALID_LIMIT"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_limit", ["0", "-1", "51", "999"])
async def test_limit_out_of_range_returns_400(
    base_settings, alive_supervisor, mock_graphiti, bad_limit
):
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)
    req = _make_request(ctx, f"query=Thuy%20Vi&limit={bad_limit}")
    resp = await search_memory_handler(req)
    assert resp.status == 400
    body = json.loads(resp.text)
    assert body["code"] == "INVALID_LIMIT"


# ─── Test C: rate limit F5 ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rate_limit_enforced_at_61st_request(
    base_settings, alive_supervisor, mock_graphiti
):
    """First _RATE_LIMIT_REQUESTS pass; 61st returns 429."""
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)

    # Mock the engine to return a trivial success so 1..60 hit 200.
    with patch(
        "janus_graph.daemon.http_search_memory_handler.engine_search_memory",
        new=AsyncMock(return_value={
            "success": True,
            "group_id": "graphiti_memory",
            "query": "Thuy Vi",
            "count": 0,
            "results": [],
        }),
    ):
        for i in range(_RATE_LIMIT_REQUESTS):
            req = _make_request(ctx, "query=Thuy%20Vi")
            resp = await search_memory_handler(req)
            assert resp.status == 200, f"request #{i + 1} should be 200, got {resp.status}"

        # 61st request → 429
        req = _make_request(ctx, "query=Thuy%20Vi")
        resp = await search_memory_handler(req)
        assert resp.status == 429
        body = json.loads(resp.text)
        assert body["code"] == "RATE_LIMITED"
        assert str(_RATE_LIMIT_REQUESTS) in body["message"]


# ─── Test J: 429 Retry-After header ────────────────────────────────────


@pytest.mark.asyncio
async def test_rate_limit_response_has_retry_after_header(
    base_settings, alive_supervisor, mock_graphiti
):
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)

    with patch(
        "janus_graph.daemon.http_search_memory_handler.engine_search_memory",
        new=AsyncMock(return_value={
            "success": True, "group_id": "graphiti_memory",
            "query": "Thuy Vi", "count": 0, "results": [],
        }),
    ):
        for _ in range(_RATE_LIMIT_REQUESTS):
            req = _make_request(ctx, "query=Thuy%20Vi")
            await search_memory_handler(req)
        req = _make_request(ctx, "query=Thuy%20Vi")
        resp = await search_memory_handler(req)
        assert resp.status == 429
        assert "Retry-After" in resp.headers
        assert int(resp.headers["Retry-After"]) == int(_RATE_LIMIT_WINDOW_SEC)


# ─── Test D: 503 SEARCH_BACKEND_DOWN when graphiti None ────────────────


@pytest.mark.asyncio
async def test_missing_graphiti_returns_503(
    base_settings, alive_supervisor
):
    ctx = _make_ctx(base_settings, alive_supervisor, graphiti=None)
    req = _make_request(ctx, "query=Thuy%20Vi")
    resp = await search_memory_handler(req)
    assert resp.status == 503
    body = json.loads(resp.text)
    assert body["code"] == "SEARCH_BACKEND_DOWN"


# ─── Test E: 503 FALKOR_DISCONNECTED when probe returns False ─────────


@pytest.mark.asyncio
async def test_falkor_down_returns_503(
    base_settings, dead_supervisor, mock_graphiti
):
    ctx = _make_ctx(base_settings, dead_supervisor, mock_graphiti)
    req = _make_request(ctx, "query=Thuy%20Vi")
    resp = await search_memory_handler(req)
    assert resp.status == 503
    body = json.loads(resp.text)
    assert body["code"] == "FALKOR_DISCONNECTED"


# ─── Test F: 200 happy path — engine delegation ─────────────────────────


@pytest.mark.asyncio
async def test_happy_path_delegates_to_engine(
    base_settings, alive_supervisor, mock_graphiti
):
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)

    engine_response = {
        "success": True,
        "group_id": "graphiti_memory",
        "query": "Thuy Vi",
        "count": 2,
        "results": [
            {"fact": "Thúy Vi là vợ của anh Maple.",
             "name": "Thúy Vi", "valid_at": "", "invalid_at": ""},
            {"fact": "Thúy Vi đồng sáng lập An Hiên Homestay.",
             "name": "Thúy Vi", "valid_at": "", "invalid_at": ""},
        ],
    }

    with patch(
        "janus_graph.daemon.http_search_memory_handler.engine_search_memory",
        new=AsyncMock(return_value=engine_response),
    ) as mock_engine:
        req = _make_request(ctx, "query=Thuy%20Vi&limit=5")
        resp = await search_memory_handler(req)

    assert resp.status == 200
    # Verify delegation: engine was called with right args
    mock_engine.assert_called_once()
    call_kwargs = mock_engine.call_args.kwargs
    assert call_kwargs["graphiti"] is mock_graphiti
    # Positional: ctx.settings, query, limit
    args = mock_engine.call_args.args
    assert args[0] is base_settings
    assert args[1] == "Thuy Vi"
    assert args[2] == 5

    body = json.loads(resp.text)
    assert body["success"] is True
    assert body["count"] == 2
    assert len(body["results"]) == 2


# ─── Test G: engine error envelope mapping ─────────────────────────────


@pytest.mark.asyncio
async def test_engine_falkor_error_maps_to_503(
    base_settings, alive_supervisor, mock_graphiti
):
    """Engine returns success=False with 'falkor' in error → 503."""
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)
    engine_response = {
        "success": False,
        "error": "FalkorDB driver disconnected at query time",
        "group_id": "graphiti_memory",
        "query": "Thuy Vi",
    }
    with patch(
        "janus_graph.daemon.http_search_memory_handler.engine_search_memory",
        new=AsyncMock(return_value=engine_response),
    ):
        req = _make_request(ctx, "query=Thuy%20Vi")
        resp = await search_memory_handler(req)
    assert resp.status == 503
    body = json.loads(resp.text)
    assert body["code"] == "FALKOR_DISCONNECTED"
    assert "falkor" in body["message"].lower()


@pytest.mark.asyncio
async def test_engine_generic_error_maps_to_500(
    base_settings, alive_supervisor, mock_graphiti
):
    """Engine returns success=False with non-falkor error → 500."""
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)
    engine_response = {
        "success": False,
        "error": "graphiti.search raised ValueError: bad config",
        "group_id": "graphiti_memory",
        "query": "Thuy Vi",
    }
    with patch(
        "janus_graph.daemon.http_search_memory_handler.engine_search_memory",
        new=AsyncMock(return_value=engine_response),
    ):
        req = _make_request(ctx, "query=Thuy%20Vi")
        resp = await search_memory_handler(req)
    assert resp.status == 500
    body = json.loads(resp.text)
    assert body["code"] == "INTERNAL_ERROR"
    assert "elapsed_ms" in body


# ─── Test H: response shape parity with stdio MCP ──────────────────────


@pytest.mark.asyncio
async def test_response_shape_matches_mcp_tool(
    base_settings, alive_supervisor, mock_graphiti
):
    """HTTP 200 body keys must be a superset of the stdio MCP
    ``search_memory`` response. Adding keys (elapsed_ms, degraded) is
    OK; missing keys would break clients that migrate from stdio.
    """
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)
    engine_response = {
        "success": True,
        "group_id": "graphiti_memory",
        "query": "Thuy Vi",
        "count": 1,
        "results": [{"fact": "test", "name": "Thúy Vi",
                     "valid_at": "", "invalid_at": ""}],
    }
    with patch(
        "janus_graph.daemon.http_search_memory_handler.engine_search_memory",
        new=AsyncMock(return_value=engine_response),
    ):
        req = _make_request(ctx, "query=Thuy%20Vi")
        resp = await search_memory_handler(req)
    body = json.loads(resp.text)

    # Stdio MCP contract keys (must all be present):
    for key in ("success", "group_id", "query", "count", "results"):
        assert key in body, f"MCP contract key {key!r} missing from HTTP body"
    # HTTP additions:
    for key in ("elapsed_ms", "degraded"):
        assert key in body, f"HTTP-only key {key!r} missing"

    # Each result row must have the same keys as the MCP path:
    for row in body["results"]:
        for key in ("fact", "name", "valid_at", "invalid_at"):
            assert key in row


# ─── Test I: F8 cancellation — aiohttp framework-level ─────────────────


@pytest.mark.asyncio
async def test_no_speculative_cancellation_code_added(
    base_settings, alive_supervisor, mock_graphiti
):
    """F8 review: aiohttp auto-cancels handler on client disconnect.
    No manual ``CancelledError`` handling should be present.

    We import the handler source and assert no ``except CancelledError``
    or ``asyncio.shield`` is wrapped around the engine call.
    """
    import inspect
    from janus_graph.daemon import http_search_memory_handler as mod

    src = inspect.getsource(mod)
    assert "except asyncio.CancelledError" not in src, (
        "F8: aiohttp already auto-cancels; manual CancelledError handling "
        "would be redundant and could swallow framework signals"
    )
    assert "asyncio.shield" not in src, (
        "F8: shield would prevent framework-level cancellation propagation"
    )


# ─── Test bonus: elapsed_ms is always a positive int ───────────────────


@pytest.mark.asyncio
async def test_elapsed_ms_is_positive(
    base_settings, alive_supervisor, mock_graphiti
):
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)
    engine_response = {
        "success": True, "group_id": "graphiti_memory",
        "query": "Thuy Vi", "count": 0, "results": [],
    }
    with patch(
        "janus_graph.daemon.http_search_memory_handler.engine_search_memory",
        new=AsyncMock(return_value=engine_response),
    ):
        req = _make_request(ctx, "query=Thuy%20Vi")
        resp = await search_memory_handler(req)
    body = json.loads(resp.text)
    assert body["elapsed_ms"] >= 0  # 0 is valid for very fast mock
    assert isinstance(body["elapsed_ms"], (int, float))


# ─── v0.5.0.1 Fix #1: 504 SEARCH_TIMEOUT on slow engine ──────────────────


@pytest.mark.asyncio
async def test_engine_call_timeout_returns_504(
    base_settings, alive_supervisor, mock_graphiti
):
    """v0.5.0.1 fix #1: bounded engine call. If engine_search_memory
    hangs longer than ``_ENGINE_CALL_TIMEOUT_SEC``, return 504
    SEARCH_TIMEOUT + Retry-After: 5 header instead of holding the
    handler open indefinitely (Falkor disconnect mid-call scenario).
    """
    import asyncio

    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)

    async def _stuck_engine(*args, **kwargs):
        # Sleep longer than the 8s handler budget. We patch the
        # ``asyncio.wait_for`` call to use a tighter timeout so the
        # test finishes quickly.
        await asyncio.sleep(60.0)
        return {"success": True, "group_id": "g", "query": "x",
                "count": 0, "results": []}

    # Patch the module constant to a tiny value so the test runs fast.
    with patch(
        "janus_graph.daemon.http_search_memory_handler.engine_search_memory",
        new=AsyncMock(side_effect=_stuck_engine),
    ), patch(
        "janus_graph.daemon.http_search_memory_handler._ENGINE_CALL_TIMEOUT_SEC",
        0.05,  # 50ms — fast test
    ):
        req = _make_request(ctx, "query=Thuy%20Vi")
        resp = await search_memory_handler(req)

    assert resp.status == 504
    body = json.loads(resp.text)
    assert body["code"] == "SEARCH_TIMEOUT"
    assert "elapsed_ms" in body
    assert resp.headers.get("Retry-After") == "5"


# ─── v0.5.0.1 Fix #4: typed engine code maps to status ──────────────────


@pytest.mark.asyncio
async def test_engine_code_falkor_disconnected_maps_to_503(
    base_settings, alive_supervisor, mock_graphiti
):
    """v0.5.0.1 fix #4: handler prefers ``code`` field over substring
    match. Engine emits ``code="FALKOR_DISCONNECTED"`` → handler
    maps to 503 even when ``error`` string doesn't contain the
    legacy magic tokens.
    """
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)
    engine_response = {
        "success": False,
        "code": "FALKOR_DISCONNECTED",
        "error": "graphiti_core transport failure (no magic word)",
        "error_type": "ConnectionError",
        "group_id": "graphiti_memory",
        "query": "Thuy Vi",
    }
    with patch(
        "janus_graph.daemon.http_search_memory_handler.engine_search_memory",
        new=AsyncMock(return_value=engine_response),
    ):
        req = _make_request(ctx, "query=Thuy%20Vi")
        resp = await search_memory_handler(req)
    assert resp.status == 503
    body = json.loads(resp.text)
    assert body["code"] == "FALKOR_DISCONNECTED"
    # Handler surfaces engine ``error`` field as ``message``
    assert body["message"] == "graphiti_core transport failure (no magic word)"


@pytest.mark.asyncio
async def test_engine_code_backend_timeout_maps_to_504(
    base_settings, alive_supervisor, mock_graphiti
):
    """v0.5.0.1 fix #4: engine code BACKEND_TIMEOUT → 504."""
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)
    engine_response = {
        "success": False,
        "code": "BACKEND_TIMEOUT",
        "error": "graphiti_search exceeded 30s",
        "error_type": "TimeoutError",
        "group_id": "graphiti_memory",
        "query": "Thuy Vi",
    }
    with patch(
        "janus_graph.daemon.http_search_memory_handler.engine_search_memory",
        new=AsyncMock(return_value=engine_response),
    ):
        req = _make_request(ctx, "query=Thuy%20Vi")
        resp = await search_memory_handler(req)
    assert resp.status == 504
    body = json.loads(resp.text)
    assert body["code"] == "SEARCH_TIMEOUT"


@pytest.mark.asyncio
async def test_engine_code_internal_error_maps_to_500(
    base_settings, alive_supervisor, mock_graphiti
):
    """v0.5.0.1 fix #4: engine code INTERNAL_ERROR → 500."""
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)
    engine_response = {
        "success": False,
        "code": "INTERNAL_ERROR",
        "error": "graphiti_core raised SomethingWeird",
        "error_type": "ValueError",
        "group_id": "graphiti_memory",
        "query": "Thuy Vi",
    }
    with patch(
        "janus_graph.daemon.http_search_memory_handler.engine_search_memory",
        new=AsyncMock(return_value=engine_response),
    ):
        req = _make_request(ctx, "query=Thuy%20Vi")
        resp = await search_memory_handler(req)
    assert resp.status == 500
    body = json.loads(resp.text)
    assert body["code"] == "INTERNAL_ERROR"


@pytest.mark.asyncio
async def test_engine_substring_fallback_still_works(
    base_settings, alive_supervisor, mock_graphiti
):
    """v0.5.0.1 fix #4: backward compat — if engine omits ``code``
    but error contains 'falkor', still map to 503. Protects MCP path
    callers (which still emit the old envelope) from regressing.
    """
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)
    engine_response = {
        "success": False,
        # NO code field — pre-v0.5.0.1 caller
        "error": "FalkorDB driver connection refused",
        "group_id": "graphiti_memory",
        "query": "Thuy Vi",
    }
    with patch(
        "janus_graph.daemon.http_search_memory_handler.engine_search_memory",
        new=AsyncMock(return_value=engine_response),
    ):
        req = _make_request(ctx, "query=Thuy%20Vi")
        resp = await search_memory_handler(req)
    assert resp.status == 503
    body = json.loads(resp.text)
    assert body["code"] == "FALKOR_DISCONNECTED"


# ─── v0.5.0.1 Fix #5: degraded field is honest (always False in 0.5.0) ──


@pytest.mark.asyncio
async def test_success_response_degraded_is_false_documented(
    base_settings, alive_supervisor, mock_graphiti
):
    """v0.5.0.1 fix #5: degraded key MUST be present on success and
    MUST be the literal False in v0.5.0 (no partial-result path yet).
    Future versions may flip this when a degraded-fallback is wired;
    until then this acts as a tripwire.
    """
    ctx = _make_ctx(base_settings, alive_supervisor, mock_graphiti)
    engine_response = {
        "success": True, "group_id": "graphiti_memory",
        "query": "Thuy Vi", "count": 1,
        "results": [{"fact": "test", "name": "Thúy Vi",
                     "valid_at": "", "invalid_at": ""}],
    }
    with patch(
        "janus_graph.daemon.http_search_memory_handler.engine_search_memory",
        new=AsyncMock(return_value=engine_response),
    ):
        req = _make_request(ctx, "query=Thuy%20Vi")
        resp = await search_memory_handler(req)
    body = json.loads(resp.text)
    assert "degraded" in body
    assert body["degraded"] is False


# ─── v0.5.0.1 Fix #6: rate limiter LRU eviction caps memory ─────────────


@pytest.mark.asyncio
async def test_rate_limiter_lru_eviction_bounds_memory():
    """v0.5.0.1 fix #6: rate limiter MUST bound total unique IPs in
    ``_hits`` dict. We instantiate a real ``_TokenBucket`` with a
    tiny ``_MAX_KEYS`` (5) and hammer 20 unique IPs through it.
    Internal ``_hits`` dict size MUST stay <= ``_MAX_KEYS + 1``
    (the +1 is the typical race-window for the most recent insert).
    """
    from janus_graph.daemon.http_search_memory_handler import (
        _TokenBucket,
    )

    bucket = _TokenBucket(max_requests=10, window_sec=60.0)
    bucket._MAX_KEYS = 5
    for i in range(20):
        assert bucket.consume(f"ip_{i}", now=float(i)) is True
    # LRU eviction: oldest IPs should be evicted from the tracking dict.
    assert len(bucket._hits) <= bucket._MAX_KEYS + 1, (
        f"_hits dict not bounded: {len(bucket._hits)} entries "
        f"(cap was {bucket._MAX_KEYS}). LRU eviction did not fire."
    )
    # Most recent IP must still be there.
    assert "ip_19" in bucket._hits, (
        "Most recent key should never be evicted (used immediately)."
    )
    # Earliest IP MUST have been evicted (it was inserted when dict
    # was empty, then 19 more came in).
    assert "ip_0" not in bucket._hits, (
        "Oldest key should be LRU-evicted when bucket is full."
    )


# ─── v0.5.0.1 Fix #2: F4 HTTP bind defaults to 127.0.0.1 ───────────────


def test_httpsettings_default_host_is_loopback():
    """v0.5.0.1 fix #2: regression test for F4. The default host MUST
    be ``127.0.0.1`` (loopback) to prevent accidental bind to a
    public interface. Replaces F4's comment-only assertion in
    http_server.py with an executable tripwire.
    """
    from janus_graph.config import HTTPSettings
    h = HTTPSettings()
    assert h.host == "127.0.0.1", (
        "F4: HTTPSettings.host default must be loopback. "
        "Changing to 0.0.0.0 exposes the daemon to the network."
    )

