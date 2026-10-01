"""Tests for daemon/http_server.py::episodes_handler (POST /episodes).

This handler was reported as an "untested hotspot" by code-review-graph
(degree 28). Verification grep over ``tests/`` confirmed 0 references, and
it is the daemon's **write path** — a silent regression here loses
episodes. Highest-value test target in the repo.

Covers every branch of the handler:
  - 503 NOT_READY      — ctx.queue_adapter is None
  - 422 INVALID_BODY   — malformed JSON / non-dict body
  - 422 INVALID_BODY   — content missing, wrong type, or whitespace-only
  - 422 INVALID_BODY   — name / group_id / source_description wrong type
  - 202                — happy path, echoes episode_id + claimed_by
  - 409 LOCK_MODE_MCP_ONLY    — adapter raises LockModeError
  - 409 DUPLICATE_EPISODE     — adapter raises EpisodeDuplicateError
  - 503 INTERNAL              — adapter raises anything else

No live Falkor / queue needed: the adapter is a MagicMock and the request
is an aiohttp ``make_mocked_request`` with a stubbed ``.json()``.

Mirrors the scaffold style of tests/test_http_search_memory_handler.py.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp.test_utils import make_mocked_request

from janus_graph.config import JanusSettings
from janus_graph.daemon.episode_queue_adapter import (
    EnqueueResult,
    EpisodeDuplicateError,
    LockModeError,
)
from janus_graph.daemon.http_server import __version__, episodes_handler
from janus_graph.daemon.lifespan import DaemonContext


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def base_settings() -> JanusSettings:
    return JanusSettings()


@pytest.fixture
def mock_adapter() -> MagicMock:
    """Queue adapter whose enqueue_guarded succeeds with a known result."""
    adapter = MagicMock()
    adapter.enqueue_guarded = AsyncMock(
        return_value=EnqueueResult(
            episode_id="ep_test1234",
            claimed_by="daemon-1",
            deduplicated=False,
        )
    )
    return adapter


def _make_ctx(settings: JanusSettings, adapter: Optional[Any]) -> DaemonContext:
    return DaemonContext(
        settings=settings,
        supervisor=MagicMock(),
        http_settings=settings.daemon.http,
        version="test",
        shutdown_event=MagicMock(),
        queue_adapter=adapter,
    )


def _post(ctx: DaemonContext, body: Any = None, raises: Optional[Exception] = None):
    """POST /episodes request with a stubbed .json().

    Stubbing .json() rather than feeding a real payload avoids aiohttp's
    payload plumbing (which requires a StreamReader) and lets us simulate
    the malformed-JSON branch precisely.
    """
    req = make_mocked_request("POST", "/episodes", app={"ctx": ctx})
    if raises is not None:
        req.json = AsyncMock(side_effect=raises)
    else:
        req.json = AsyncMock(return_value=body)
    return req


async def _json_of(resp) -> Any:
    return __import__("json").loads(resp.body.decode())


# ─── 503 NOT_READY ──────────────────────────────────────────────────────


async def test_missing_queue_adapter_returns_503(base_settings):
    """A booting daemon has queue_adapter=None; must not 500."""
    ctx = _make_ctx(base_settings, adapter=None)
    resp = await episodes_handler(_post(ctx, {"content": "hi"}))
    assert resp.status == 503
    body = await _json_of(resp)
    assert body["code"] == "NOT_READY"
    assert "queue adapter" in body["message"]


# ─── 422 body-shape validation ──────────────────────────────────────────


async def test_malformed_json_returns_422(base_settings, mock_adapter):
    ctx = _make_ctx(base_settings, mock_adapter)
    req = _post(ctx, raises=ValueError("Expecting value: line 1 column 1"))
    resp = await episodes_handler(req)
    assert resp.status == 422
    body = await _json_of(resp)
    assert body["code"] == "INVALID_BODY"
    assert "malformed JSON" in body["message"]
    mock_adapter.enqueue_guarded.assert_not_called()


async def test_non_dict_body_returns_422(base_settings, mock_adapter):
    """A JSON array is valid JSON but the wrong shape → 422, not 400."""
    ctx = _make_ctx(base_settings, mock_adapter)
    resp = await episodes_handler(_post(ctx, ["a", "b"]))
    assert resp.status == 422
    assert (await _json_of(resp))["code"] == "INVALID_BODY"
    mock_adapter.enqueue_guarded.assert_not_called()


@pytest.mark.parametrize(
    "bad_content",
    [None, 123, "", "   ", " " + chr(10) + chr(9) + " "],
    ids=["missing", "int", "empty", "spaces", "whitespace"],
)
async def test_invalid_content_returns_422(base_settings, mock_adapter, bad_content):
    ctx = _make_ctx(base_settings, mock_adapter)
    resp = await episodes_handler(_post(ctx, {"content": bad_content}))
    assert resp.status == 422
    body = await _json_of(resp)
    assert body["code"] == "INVALID_BODY"
    assert "content" in body["message"]
    mock_adapter.enqueue_guarded.assert_not_called()


async def test_non_string_name_returns_422(base_settings, mock_adapter):
    ctx = _make_ctx(base_settings, mock_adapter)
    resp = await episodes_handler(_post(ctx, {"content": "ok", "name": 42}))
    assert resp.status == 422
    assert "name" in (await _json_of(resp))["message"]
    mock_adapter.enqueue_guarded.assert_not_called()


@pytest.mark.parametrize("bad_group", ["", 7, None], ids=["empty", "int", "null"])
async def test_invalid_group_id_returns_422(base_settings, mock_adapter, bad_group):
    """group_id defaults only when the key is absent; explicit bad values 422."""
    ctx = _make_ctx(base_settings, mock_adapter)
    resp = await episodes_handler(_post(ctx, {"content": "ok", "group_id": bad_group}))
    assert resp.status == 422
    assert "group_id" in (await _json_of(resp))["message"]
    mock_adapter.enqueue_guarded.assert_not_called()


async def test_non_string_source_description_returns_422(base_settings, mock_adapter):
    ctx = _make_ctx(base_settings, mock_adapter)
    resp = await episodes_handler(_post(ctx, {"content": "ok", "source_description": 99}))
    assert resp.status == 422
    assert "source_description" in (await _json_of(resp))["message"]
    mock_adapter.enqueue_guarded.assert_not_called()


# ─── 202 happy path ─────────────────────────────────────────────────────


async def test_happy_path_returns_202_with_envelope(base_settings, mock_adapter):
    ctx = _make_ctx(base_settings, mock_adapter)
    resp = await episodes_handler(
        _post(ctx, {"content": "Maple fixed the MMR floor", "name": "ep_abc"})
    )
    assert resp.status == 202
    body = await _json_of(resp)
    assert body == {
        "episode_id": "ep_test1234",
        "claimed_by": "daemon-1",
        "deduplicated": False,
        "daemon_version": __version__,
    }


async def test_happy_path_forwards_all_fields_to_adapter(base_settings, mock_adapter):
    """The handler must not silently drop optional fields on the way down."""
    ctx = _make_ctx(base_settings, mock_adapter)
    resp = await episodes_handler(
        _post(
            ctx,
            {
                "content": "c",
                "name": "n",
                "group_id": "tenant_x",
                "source_description": "unit_test",
            },
        )
    )
    assert resp.status == 202
    mock_adapter.enqueue_guarded.assert_awaited_once_with(
        content="c",
        group_id="tenant_x",
        name="n",
        source_description="unit_test",
    )


async def test_optional_fields_default_when_absent(base_settings, mock_adapter):
    """Omitting name/group_id/source_description must use handler defaults,
    NOT the adapter's signature defaults — this locks the contract."""
    ctx = _make_ctx(base_settings, mock_adapter)
    resp = await episodes_handler(_post(ctx, {"content": "c"}))
    assert resp.status == 202
    mock_adapter.enqueue_guarded.assert_awaited_once_with(
        content="c",
        group_id="graphiti_memory",
        name=None,
        source_description="daemon_http",
    )


async def test_deduplicated_flag_is_passed_through(base_settings):
    """A dedup hit that resolves normally surfaces deduplicated=True, 202."""
    ctx = _make_ctx(base_settings, adapter=MagicMock())
    ctx.queue_adapter.enqueue_guarded = AsyncMock(
        return_value=EnqueueResult(episode_id="ep_dup", claimed_by=None, deduplicated=True)
    )
    resp = await episodes_handler(_post(ctx, {"content": "c"}))
    assert resp.status == 202
    body = await _json_of(resp)
    assert body["deduplicated"] is True
    assert body["claimed_by"] is None


# ─── 409 lock mode ──────────────────────────────────────────────────────


async def test_lock_mode_error_returns_409(base_settings):
    ctx = _make_ctx(base_settings, adapter=MagicMock())
    ctx.queue_adapter.enqueue_guarded = AsyncMock(
        side_effect=LockModeError("daemon writes disabled (mode=mcp_only)")
    )
    resp = await episodes_handler(_post(ctx, {"content": "c"}))
    assert resp.status == 409
    body = await _json_of(resp)
    assert body["code"] == "LOCK_MODE_MCP_ONLY"
    assert "mcp_only" in body["message"]


# ─── 409 duplicate ──────────────────────────────────────────────────────


async def test_duplicate_episode_returns_409_with_existing_id(base_settings):
    ctx = _make_ctx(base_settings, adapter=MagicMock())
    ctx.queue_adapter.enqueue_guarded = AsyncMock(
        side_effect=EpisodeDuplicateError("ep_existing99")
    )
    resp = await episodes_handler(_post(ctx, {"content": "c"}))
    assert resp.status == 409
    body = await _json_of(resp)
    assert body["code"] == "DUPLICATE_EPISODE"
    assert body["existing_episode_id"] == "ep_existing99"


# ─── 503 INTERNAL (defensive) ───────────────────────────────────────────


async def test_unexpected_adapter_error_returns_503(base_settings):
    ctx = _make_ctx(base_settings, adapter=MagicMock())
    ctx.queue_adapter.enqueue_guarded = AsyncMock(side_effect=RuntimeError("sqlite is locked"))
    resp = await episodes_handler(_post(ctx, {"content": "c"}))
    assert resp.status == 503
    body = await _json_of(resp)
    assert body["code"] == "INTERNAL"
    assert "sqlite is locked" in body["message"]
