"""Tests for daemon/http_server.py::search_graph_handler (GET /search/graph).

Second "untested hotspot" from code-review-graph (degree 24, 14 branches -
the most branch-dense untested function in the repo). The handler is a
thin envelope over ``SearchEngine.search`` but owns real logic of its own:

  - GET query-string parsing (comma lists, int coercion, bool coercion)
  - error-code -> HTTP-status mapping (400/404/422/503/504)
  - defaulting ``query`` to the first seed entity
  - injecting ``daemon_version`` into the 200 envelope

``tests/test_search_graph.py`` already covers the layer BELOW this one
(cosine, validate_request, mmr, parse_falkor_rows) but never invokes the
handler, so none of the mapping below was covered.

The engine is a MagicMock - no live Falkor / llama-server required.
"""

from __future__ import annotations

import json as _json
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp.test_utils import make_mocked_request

from janus_graph.config import JanusSettings
from janus_graph.daemon.http_server import __version__, search_graph_handler
from janus_graph.daemon.lifespan import DaemonContext
from janus_graph.daemon.search_engine import (
    FactRow,
    SearchBackendError,
    SearchResponse,
    SearchValidationError,
)


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def base_settings() -> JanusSettings:
    return JanusSettings()


@pytest.fixture
def ok_result() -> SearchResponse:
    return SearchResponse(
        results=[
            FactRow(
                fact="FalkorDB indexes the knowledge graph",
                source_entity="FalkorDB",
                target_entity="knowledge graph",
                edge_type="RELATES_TO",
                hop_distance=1,
                path=["FalkorDB", "knowledge graph"],
                cosine=0.87,
                invalid_at=None,
            )
        ],
        count=1,
        elapsed_ms=42,
        degraded=False,
        warnings=[],
    )


@pytest.fixture
def mock_engine(ok_result) -> MagicMock:
    engine = MagicMock()
    engine.search = AsyncMock(return_value=ok_result)
    return engine


def _make_ctx(
    settings: JanusSettings,
    engine: Optional[Any],
    search_graph_enabled: bool = True,
) -> DaemonContext:
    ctx = DaemonContext(
        settings=settings,
        supervisor=MagicMock(),
        http_settings=settings.daemon.http,
        version="test",
        shutdown_event=MagicMock(),
        search_engine=engine,
    )
    ctx.settings.daemon.search_graph.enabled = search_graph_enabled
    return ctx


def _get(ctx: DaemonContext, qs: str = ""):
    return make_mocked_request(
        "GET", "/search/graph" + (("?" + qs) if qs else ""), app={"ctx": ctx}
    )


async def _json_of(resp) -> Any:
    return _json.loads(resp.body.decode())


# ─── 503 guards ─────────────────────────────────────────────────────────


async def test_missing_search_engine_returns_503(base_settings):
    ctx = _make_ctx(base_settings, engine=None)
    resp = await search_graph_handler(_get(ctx, "seed_entities=a&max_hops=1"))
    assert resp.status == 503
    body = await _json_of(resp)
    assert body["code"] == "NOT_READY"
    assert "search engine" in body["message"]


async def test_disabled_in_config_returns_503(base_settings, mock_engine):
    """enabled=false must short-circuit BEFORE the engine is touched."""
    ctx = _make_ctx(base_settings, mock_engine, search_graph_enabled=False)
    resp = await search_graph_handler(_get(ctx, "seed_entities=a&max_hops=1"))
    assert resp.status == 503
    body = await _json_of(resp)
    assert body["code"] == "DISABLED"
    assert "search_graph disabled" in body["message"]
    mock_engine.search.assert_not_called()


# ─── GET query-string parsing ───────────────────────────────────────────


async def test_get_comma_separated_seeds_and_ints(base_settings, mock_engine):
    ctx = _make_ctx(base_settings, mock_engine)
    resp = await search_graph_handler(
        _get(ctx, "seed_entities=alpha,beta,gamma&max_hops=2&limit=7")
    )
    assert resp.status == 200
    payload = mock_engine.search.await_args.args[0]
    assert payload["seed_entities"] == ["alpha", "beta", "gamma"]
    assert payload["max_hops"] == 2
    assert payload["limit"] == 7


async def test_get_trims_whitespace_and_drops_empty_seeds(base_settings, mock_engine):
    """?seed_entities= a , , b  -> ['a','b'] (split/strip/filter chain)."""
    ctx = _make_ctx(base_settings, mock_engine)
    resp = await search_graph_handler(
        _get(ctx, "seed_entities= a , , b &max_hops=1")
    )
    assert resp.status == 200
    assert mock_engine.search.await_args.args[0]["seed_entities"] == ["a", "b"]


async def test_get_edge_types_parsed_as_comma_list(base_settings, mock_engine):
    ctx = _make_ctx(base_settings, mock_engine)
    resp = await search_graph_handler(
        _get(ctx, "seed_entities=a&max_hops=1&edge_types=RELATES_TO,CALLS")
    )
    assert resp.status == 200
    assert mock_engine.search.await_args.args[0]["edge_types"] == [
        "RELATES_TO",
        "CALLS",
    ]


@pytest.mark.parametrize("bad", ["abc", "1.5", "", "1e3"], ids=["word", "float", "empty", "sci"])
async def test_get_non_int_max_hops_returns_400(base_settings, mock_engine, bad):
    """int() raises ValueError -> 400, and the engine is never called."""
    ctx = _make_ctx(base_settings, mock_engine)
    resp = await search_graph_handler(
        _get(ctx, "seed_entities=a&max_hops=" + bad)
    )
    assert resp.status == 400
    body = await _json_of(resp)
    assert body["code"] == "INVALID_BODY"
    assert "max_hops must be int" in body["message"]
    mock_engine.search.assert_not_called()


async def test_get_non_int_limit_returns_400(base_settings, mock_engine):
    ctx = _make_ctx(base_settings, mock_engine)
    resp = await search_graph_handler(_get(ctx, "seed_entities=a&max_hops=1&limit=xyz"))
    assert resp.status == 400
    assert "limit must be int" in (await _json_of(resp))["message"]
    mock_engine.search.assert_not_called()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("maybe", False),
    ],
)
async def test_get_include_episodes_bool_coercion(
    base_settings, mock_engine, raw, expected
):
    """Only 1/true/yes (case-insensitive) are truthy; everything else False."""
    ctx = _make_ctx(base_settings, mock_engine)
    resp = await search_graph_handler(
        _get(ctx, "seed_entities=a&max_hops=1&include_episodes=" + raw)
    )
    assert resp.status == 200
    assert mock_engine.search.await_args.args[0]["include_episodes"] is expected


async def test_get_passes_through_float_and_string_params(base_settings, mock_engine):
    """min_cosine / mmr_lambda / query stay as raw strings for the engine
    to validate - the handler must NOT coerce or clamp them."""
    ctx = _make_ctx(base_settings, mock_engine)
    resp = await search_graph_handler(
        _get(ctx, "seed_entities=a&max_hops=1&min_cosine=0.9&mmr_lambda=0.3&query=hello")
    )
    assert resp.status == 200
    payload = mock_engine.search.await_args.args[0]
    assert payload["min_cosine"] == "0.9"
    assert payload["mmr_lambda"] == "0.3"
    assert payload["query"] == "hello"


async def test_get_absent_params_are_omitted_not_nulled(base_settings, mock_engine):
    """Unset keys must be absent from the payload, not present-as-None -
    validate_request distinguishes 'missing' from 'None'. ('query' is
    excluded: it is intentionally derived from the first seed.)"""
    ctx = _make_ctx(base_settings, mock_engine)
    resp = await search_graph_handler(_get(ctx, "seed_entities=a&max_hops=1"))
    assert resp.status == 200
    payload = mock_engine.search.await_args.args[0]
    for key in ("min_cosine", "mmr_lambda", "limit", "include_episodes",
                "edge_types"):
        assert key not in payload, key
    assert payload["seed_entities"] == ["a"]
    assert payload["max_hops"] == 1


# ─── default query derivation ───────────────────────────────────────────


async def test_query_defaults_to_first_seed(base_settings, mock_engine):
    """This default drives the cosine step of the pipeline - a silent drop
    would return zero facts with no error."""
    ctx = _make_ctx(base_settings, mock_engine)
    resp = await search_graph_handler(
        _get(ctx, "seed_entities=alpha,beta&max_hops=1")
    )
    assert resp.status == 200
    assert mock_engine.search.await_args.args[0]["query"] == "alpha"


async def test_explicit_query_wins_over_seed_default(base_settings, mock_engine):
    ctx = _make_ctx(base_settings, mock_engine)
    resp = await search_graph_handler(
        _get(ctx, "seed_entities=alpha,beta&max_hops=1&query=explicit")
    )
    assert resp.status == 200
    assert mock_engine.search.await_args.args[0]["query"] == "explicit"


async def test_no_query_injection_when_seeds_absent(base_settings, mock_engine):
    """Empty/absent seeds must not fabricate a query - the engine's own
    validate_request owns the EMPTY_SEEDS error."""
    ctx = _make_ctx(base_settings, mock_engine)
    resp = await search_graph_handler(_get(ctx, "max_hops=1"))
    assert resp.status == 200
    assert "query" not in mock_engine.search.await_args.args[0]


# ─── error-code -> HTTP status mapping ───────────────────────────────────


async def test_min_cosine_out_of_range_maps_to_422(base_settings):
    ctx = _make_ctx(base_settings, _engine_raising(
        SearchValidationError("MIN_COSINE_OUT_OF_RANGE", "must be 0..1")))
    resp = await search_graph_handler(_get(ctx, "seed_entities=a&max_hops=1"))
    assert resp.status == 422
    body = await _json_of(resp)
    assert body["code"] == "MIN_COSINE_OUT_OF_RANGE"
    assert body["message"] == "must be 0..1"


async def test_other_validation_error_maps_to_400(base_settings):
    ctx = _make_ctx(base_settings, _engine_raising(
        SearchValidationError("INVALID_HOPS", "hops must be 1..3")))
    resp = await search_graph_handler(_get(ctx, "seed_entities=a&max_hops=9"))
    assert resp.status == 400
    assert (await _json_of(resp))["code"] == "INVALID_HOPS"


async def test_seed_not_found_maps_to_404(base_settings):
    ctx = _make_ctx(base_settings, _engine_raising(
        SearchBackendError("SEED_NOT_FOUND", "no entity alpha")))
    resp = await search_graph_handler(_get(ctx, "seed_entities=alpha&max_hops=1"))
    assert resp.status == 404
    assert (await _json_of(resp))["code"] == "SEED_NOT_FOUND"


async def test_traversal_timeout_maps_to_504(base_settings):
    ctx = _make_ctx(base_settings, _engine_raising(
        SearchBackendError("TRAVERSAL_TIMEOUT", "exceeded 2.0s")))
    resp = await search_graph_handler(_get(ctx, "seed_entities=a&max_hops=3"))
    assert resp.status == 504
    assert (await _json_of(resp))["code"] == "TRAVERSAL_TIMEOUT"


async def test_falkor_down_maps_to_503(base_settings):
    ctx = _make_ctx(base_settings, _engine_raising(
        SearchBackendError("FALKOR_DOWN", "cannot reach :6379")))
    resp = await search_graph_handler(_get(ctx, "seed_entities=a&max_hops=1"))
    assert resp.status == 503
    assert (await _json_of(resp))["code"] == "FALKOR_DOWN"


async def test_unmapped_backend_error_code_defaults_to_503(base_settings):
    """Unknown backend code must still degrade to 503, never leak a 200."""
    ctx = _make_ctx(base_settings, _engine_raising(
        SearchBackendError("SOMETHING_NEW", "unmapped")))
    resp = await search_graph_handler(_get(ctx, "seed_entities=a&max_hops=1"))
    assert resp.status == 503
    assert (await _json_of(resp))["code"] == "SOMETHING_NEW"


async def test_unexpected_exception_maps_to_503_internal(base_settings):
    ctx = _make_ctx(base_settings, _engine_raising(RuntimeError("boom")))
    resp = await search_graph_handler(_get(ctx, "seed_entities=a&max_hops=1"))
    assert resp.status == 503
    body = await _json_of(resp)
    assert body["code"] == "INTERNAL"
    assert "boom" in body["message"]


# ─── 200 envelope ───────────────────────────────────────────────────────


async def test_success_envelope_shape(base_settings, mock_engine):
    """Locks the exact 200 body: engine to_dict + injected daemon_version."""
    ctx = _make_ctx(base_settings, mock_engine)
    resp = await search_graph_handler(_get(ctx, "seed_entities=FalkorDB&max_hops=1"))
    assert resp.status == 200
    body = await _json_of(resp)
    assert set(body) == {
        "results", "count", "elapsed_ms", "degraded", "warnings", "daemon_version",
    }
    assert body["count"] == 1
    assert body["elapsed_ms"] == 42
    assert body["degraded"] is False
    assert body["warnings"] == []
    assert body["daemon_version"] == __version__
    assert body["results"][0]["fact"] == "FalkorDB indexes the knowledge graph"
    assert body["results"][0]["cosine"] == 0.87


async def test_degraded_and_warnings_are_surfaced(base_settings):
    """degraded=true + warnings must reach the client verbatim - this is
    how a Falkor/embed partial failure is reported to the hook."""
    engine = MagicMock()
    engine.search = AsyncMock(
        return_value=SearchResponse(
            results=[], count=0, elapsed_ms=7, degraded=True,
            warnings=["embedding degraded", "2 seeds missing"],
        )
    )
    ctx = _make_ctx(base_settings, engine)
    resp = await search_graph_handler(_get(ctx, "seed_entities=a&max_hops=1"))
    assert resp.status == 200
    body = await _json_of(resp)
    assert body["degraded"] is True
    assert body["warnings"] == ["embedding degraded", "2 seeds missing"]


# ─── helper ─────────────────────────────────────────────────────────────


def _engine_raising(exc: Exception) -> MagicMock:
    engine = MagicMock()
    engine.search = AsyncMock(side_effect=exc)
    return engine
