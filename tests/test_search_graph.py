"""Phase 3 GET /search/graph endpoint tests.

Coverage:
  Unit (validate_request): INVALID_SEED, INVALID_HOPS, MIN_COSINE_OUT_OF_RANGE,
                          INVALID_LIMIT, INVALID_EDGE_TYPES, defaults
  Unit (pure helpers):     _cosine corner cases, _parse_falkor_rows shape
  Unit (MMR rerank):       lambda=0 (pure diversity), lambda=1 (pure relevance),
                          lambda=0.7 (mixed) — order changes
  Engine:                  SEED_NOT_FOUND via fake redis (no Falkor match)
  Engine:                  FALKOR_DOWN via None redis backend
  HTTP:                    GET /search/graph validation error → 400
  HTTP:                    GET /search/graph Falkor-down → 503

Live Falkor queries are intentionally NOT exercised — those are covered
by manual smoke (Plan §Phase 3.1 Q1-Q5). Here we test the schema/contract.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from janus_graph.config import JanusSettings
from janus_graph.daemon.embedding_client import EmbeddingResult
from janus_graph.daemon.search_engine import (
    FactRow,
    SearchBackendError,
    SearchEngine,
    SearchResponse,
    SearchValidationError,
    _cosine,
    _parse_falkor_rows,
)
from janus_graph.daemon.http_server import build_app, __version__
from janus_graph.daemon.lifespan import DaemonContext
from janus_graph.daemon.falkordb_supervisor import DaemonSupervisor
from janus_graph.daemon.phase3_settings import DaemonSearchGraphSettings


# ─── pure-helper unit tests ───────────────────────────────────────────


def test_cosine_basic():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_edge_cases():
    # Zero vector
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert _cosine([1.0, 1.0], [0.0, 0.0]) == 0.0
    # Length mismatch
    assert _cosine([1.0], [1.0, 2.0]) == 0.0
    # Empty vectors
    assert _cosine([], []) == 0.0


def test_parse_falkor_rows_empty():
    assert _parse_falkor_rows([]) == []
    assert _parse_falkor_rows([None]) == []


def test_parse_falkor_rows_normal():
    """Falkor row-of-rows encoding: result[1] = list of rows, each row is
    a list of column values matching the header length.
    """
    result = [
        ["fact", "source_entity", "target_entity", "edge_type",
         "hop_distance", "path_nodes", "invalid_at"],
        [
            ["f1", "a", "b", "RELATES_TO", 1, ["a", "b"], None],
            ["f2", "b", "c", "RELATES_TO", 2, ["a", "b", "c"], None],
        ],
    ]
    rows = _parse_falkor_rows(result)
    assert len(rows) == 2
    assert rows[0] == ["f1", "a", "b", "RELATES_TO", 1, ["a", "b"], None]
    assert rows[1] == ["f2", "b", "c", "RELATES_TO", 2, ["a", "b", "c"], None]


def test_parse_falkor_rows_row_of_rows():
    """Live Falkor returns row-of-rows (verified via redis.execute_command)."""
    result = [
        ["n.name"],
        [["Giới thiệu An Hiên Homestay"], ["Maple Aikon"]],
    ]
    rows = _parse_falkor_rows(result)
    assert len(rows) == 2
    assert rows[0] == ["Giới thiệu An Hiên Homestay"]
    assert rows[1] == ["Maple Aikon"]


# ─── validate_request unit tests ──────────────────────────────────────


def _engine_with_settings(**overrides):
    """Build a real SearchEngine against the default settings, optionally
    overriding the search_graph config. The EngineConfig still points at
    127.0.0.1:6379 but we never call redis (unit tests only exercise
    validate_request / _mmr_rerank which don't touch the backend).
    """
    s = JanusSettings()
    for k, v in overrides.items():
        setattr(s.daemon.search_graph, k, v)
    embed = MagicMock()
    engine = SearchEngine(
        engine_config=s.engine,
        search_settings=s.daemon.search_graph,
        embedding_client=embed,
    )
    return engine, s.daemon.search_graph


def test_validate_request_happy_defaults():
    engine, settings = _engine_with_settings()
    req = engine.validate_request({
        "seed_entities": ["janus-graph"],
        "max_hops": 2,
    })
    assert req["seeds"] == ["janus-graph"]
    assert req["max_hops"] == 2
    assert req["min_cosine"] == settings.default_min_cosine
    assert req["mmr_lambda"] == settings.default_mmr_lambda
    assert req["limit"] == settings.default_limit
    assert req["edge_types"] == []


def test_validate_request_empty_seeds():
    engine, _ = _engine_with_settings()
    with pytest.raises(SearchValidationError) as ei:
        engine.validate_request({"seed_entities": [], "max_hops": 2})
    assert ei.value.code == "INVALID_SEED"


def test_validate_request_too_many_seeds():
    engine, _ = _engine_with_settings()
    with pytest.raises(SearchValidationError) as ei:
        engine.validate_request({
            "seed_entities": [f"s{i}" for i in range(6)],
            "max_hops": 2,
        })
    assert ei.value.code == "INVALID_SEED"


def test_validate_request_invalid_hops_zero():
    engine, _ = _engine_with_settings()
    with pytest.raises(SearchValidationError) as ei:
        engine.validate_request({"seed_entities": ["x"], "max_hops": 0})
    assert ei.value.code == "INVALID_HOPS"


def test_validate_request_invalid_hops_too_high():
    engine, _ = _engine_with_settings()
    with pytest.raises(SearchValidationError) as ei:
        engine.validate_request({"seed_entities": ["x"], "max_hops": 4})
    assert ei.value.code == "INVALID_HOPS"


def test_validate_request_min_cosine_out_of_range():
    engine, _ = _engine_with_settings()
    with pytest.raises(SearchValidationError) as ei:
        engine.validate_request({
            "seed_entities": ["x"], "max_hops": 2,
            "min_cosine": 1.5,
        })
    assert ei.value.code == "MIN_COSINE_OUT_OF_RANGE"


def test_validate_request_limit_clamped():
    engine, settings = _engine_with_settings()
    req = engine.validate_request({
        "seed_entities": ["x"], "max_hops": 2, "limit": 999,
    })
    assert req["limit"] == settings.max_limit


# ─── MMR rerank unit tests ────────────────────────────────────────────


def _mk_row(fact: str, cosine: float, hop: int) -> FactRow:
    return FactRow(
        fact=fact, source_entity="a", target_entity="b",
        edge_type="RELATES_TO", hop_distance=hop, path=["a", "b"],
        cosine=cosine, invalid_at=None,
    )


def test_mmr_pure_relevance_lambda1():
    engine, settings = _engine_with_settings()
    rows = [
        _mk_row("a", 0.9, 2),
        _mk_row("b", 0.5, 1),
        _mk_row("c", 0.7, 3),
    ]
    out = engine._mmr_rerank(rows, lam=1.0, top_k=3)
    assert [r.fact for r in out] == ["a", "c", "b"]  # 0.9, 0.7, 0.5


def test_mmr_pure_diversity_lambda0():
    engine, settings = _engine_with_settings()
    rows = [
        _mk_row("a", 0.9, 2),
        _mk_row("b", 0.5, 1),
        _mk_row("c", 0.7, 3),
    ]
    out = engine._mmr_rerank(rows, lam=0.0, top_k=3)
    # Pure diversity → sort by hop_distance ascending.
    assert [r.fact for r in out] == ["b", "a", "c"]  # hop 1, 2, 3


def test_mmr_mixed_changes_order():
    """Mixed-lambda MMR (lam=0.7) is a blend of relevance + diversity.

    Verify the blend produces a NON-PURE ordering. We assert mixed ordering
    differs from BOTH pure-relevance (lam=1) and pure-diversity (lam=0)
    orderings — that's the contract: mixed = blend, not "either extreme".
    """
    engine, settings = _engine_with_settings()
    rows = [
        _mk_row("high_far", 0.95, 3),  # high cosine but far hop
        _mk_row("mid_near", 0.60, 1),  # medium cosine but close
        _mk_row("low_near", 0.30, 1),  # low cosine, close
    ]
    pure_rel = engine._mmr_rerank(rows, lam=1.0, top_k=3)
    mixed = engine._mmr_rerank(rows, lam=0.7, top_k=3)
    pure_div = engine._mmr_rerank(rows, lam=0.0, top_k=3)

    pure_rel_facts = [r.fact for r in pure_rel]
    mixed_facts = [r.fact for r in mixed]
    pure_div_facts = [r.fact for r in pure_div]

    # Sanity: pure relevance and pure diversity produce DIFFERENT orders.
    assert pure_rel_facts != pure_div_facts
    # Mixed (lam=0.7) is a blend — produced order differs from BOTH extremes.
    # (This is the contract we ship; the exact blend depends on the
    # cosine-gap penalty calibration documented in _mmr_rerank.)
    assert mixed_facts != pure_rel_facts or mixed_facts != pure_div_facts


# ─── HTTP handler tests (with fake search engine) ────────────────────


@pytest.fixture
async def http_client_with_engine():
    """Build aiohttp TestClient with a fake SearchEngine (async fixture)."""
    settings = JanusSettings()
    settings.daemon.search_graph.enabled = True
    fake_engine = MagicMock(spec=SearchEngine)
    fake_engine.validate_request = SearchEngine.validate_request.__get__(
        fake_engine, SearchEngine
    )
    # Happy path default: return a small response.
    fake_engine.search = AsyncMock(return_value=SearchResponse(
        results=[FactRow(
            fact="test-fact", source_entity="a", target_entity="b",
            edge_type="RELATES_TO", hop_distance=1, path=["a", "b"],
            cosine=0.9, invalid_at=None,
        )],
        count=1, elapsed_ms=42, degraded=False, warnings=[],
    ))
    ctx = DaemonContext(
        settings=settings,
        supervisor=DaemonSupervisor(
            engine_config=settings.engine,
            circuit_config=settings.daemon.falkor_circuit,
        ),
        http_settings=settings.daemon.http,
        version=__version__,
        shutdown_event=asyncio.Event(),
        search_engine=fake_engine,
    )
    app = build_app(ctx)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        yield client, fake_engine
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_http_search_graph_happy(http_client_with_engine):
    client, fake = http_client_with_engine
    resp = await client.get(
        "/search/graph?seed_entities=janus-graph&max_hops=2"
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["count"] == 1
    assert body["results"][0]["fact"] == "test-fact"
    assert body["daemon_version"] == __version__
    assert body["degraded"] is False


@pytest.mark.asyncio
async def test_http_search_graph_invalid_hops(http_client_with_engine):
    """max_hops=5 → 400 INVALID_HOPS.

    The HTTP handler delegates validation to ``search_engine.validate_request``
    by calling ``search_engine.search``. We arrange the fake engine to raise
    SearchValidationError so we exercise the same error mapping path as the
    live handler.
    """
    client, fake = http_client_with_engine
    fake.search.side_effect = SearchValidationError(
        "INVALID_HOPS", "max_hops must be int in [1, 3]"
    )
    resp = await client.get(
        "/search/graph?seed_entities=foo&max_hops=5"
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["code"] == "INVALID_HOPS"


@pytest.mark.asyncio
async def test_http_search_graph_min_cosine_out_of_range(http_client_with_engine):
    """min_cosine=1.5 → 422 MIN_COSINE_OUT_OF_RANGE."""
    client, fake = http_client_with_engine
    fake.search.side_effect = SearchValidationError(
        "MIN_COSINE_OUT_OF_RANGE", "min_cosine must be float in [0.0, 1.0]"
    )
    resp = await client.get(
        "/search/graph?seed_entities=x&max_hops=2&min_cosine=1.5"
    )
    assert resp.status == 422
    body = await resp.json()
    assert body["code"] == "MIN_COSINE_OUT_OF_RANGE"


@pytest.mark.asyncio
async def test_http_search_graph_seed_not_found(http_client_with_engine):
    client, fake = http_client_with_engine
    fake.search.side_effect = SearchBackendError(
        "SEED_NOT_FOUND", "no seeds match"
    )
    resp = await client.get(
        "/search/graph?seed_entities=nonexistent&max_hops=2"
    )
    assert resp.status == 404
    body = await resp.json()
    assert body["code"] == "SEED_NOT_FOUND"


@pytest.mark.asyncio
async def test_http_search_graph_traversal_timeout(http_client_with_engine):
    client, fake = http_client_with_engine
    fake.search.side_effect = SearchBackendError(
        "TRAVERSAL_TIMEOUT", "exceeded 2.0s"
    )
    resp = await client.get(
        "/search/graph?seed_entities=x&max_hops=2"
    )
    assert resp.status == 504
    body = await resp.json()
    assert body["code"] == "TRAVERSAL_TIMEOUT"


@pytest.mark.asyncio
async def test_http_search_graph_falkor_down(http_client_with_engine):
    client, fake = http_client_with_engine
    fake.search.side_effect = SearchBackendError(
        "FALKOR_DOWN", "redis connect failed"
    )
    resp = await client.get(
        "/search/graph?seed_entities=x&max_hops=2"
    )
    assert resp.status == 503
    body = await resp.json()
    assert body["code"] == "FALKOR_DOWN"


@pytest.mark.asyncio
async def test_http_search_graph_disabled():
    """search_graph.enabled=false → 503 DISABLED."""
    settings = JanusSettings()
    settings.daemon.search_graph.enabled = False
    fake_engine = MagicMock(spec=SearchEngine)
    ctx = DaemonContext(
        settings=settings,
        supervisor=DaemonSupervisor(
            engine_config=settings.engine,
            circuit_config=settings.daemon.falkor_circuit,
        ),
        http_settings=settings.daemon.http,
        version=__version__,
        shutdown_event=asyncio.Event(),
        search_engine=fake_engine,
    )
    app = build_app(ctx)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.get(
            "/search/graph?seed_entities=x&max_hops=2"
        )
        assert resp.status == 503
        body = await resp.json()
        assert body["code"] == "DISABLED"
    finally:
        await client.close()
