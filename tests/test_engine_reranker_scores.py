"""``search_memory`` reranker scores + diagnostics (plan wave 2, 2026-10-16).

Why the plumbing is this defensive: the vendor returns either a
``SearchResults`` object (``graphiti_core/search/search_config.py:121``) or,
on some pinned builds, a bare list of edges — the facade already carries a
``getattr(..., "edges", search_results)`` fallback for that. The scores live
in ``SearchResults.edge_reranker_scores``, which is index-parallel with
``edges`` BEFORE the cosine gate runs (vendor ``search.py:461`` returns
``reranked_edges[:limit], edge_scores[:limit]``). The gate then drops edges,
so a ``zip`` over two lists is off by N rows the moment the gate fires.

Scores are therefore matched by ``edge.uuid``, and a missing/short score list
degrades to ``None`` rather than a plausible-looking ``0.0``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from janus_graph.config import JanusSettings
from janus_graph.engine.search_memory import search_memory


def _edge(uuid: str, fact: str):
    e = MagicMock()
    e.uuid = uuid
    e.fact = fact
    e.name = ""
    e.valid_at = ""
    e.invalid_at = None
    return e


def _install(monkeypatch, result, *, edges_before_gate=None):
    """Patch ``graphiti_search`` to return ``result``; return the mock."""
    from janus_graph.engine import search_memory as engine_module

    async def _fake(**kwargs):
        return result

    mock = MagicMock(return_value=result)
    mock.call_args = None
    monkeypatch.setattr(engine_module, "graphiti_search", _fake)
    monkeypatch.setattr(engine_module, "resolve_cosine_gate_min", lambda cfg: None, raising=False)
    g = MagicMock()
    g.clients = MagicMock()
    g.clients.embedder.create = AsyncMock(return_value=None)
    return g


@pytest.fixture
def base_settings(temp_dir) -> JanusSettings:
    s = JanusSettings()
    s.pipeline.queue_db_path = str(temp_dir / "episodes.db")
    return s


# ── AC1/AC2: scores are present, correct per row, and ordered ───────────────


@pytest.mark.asyncio
async def test_scores_are_present_and_decrease(base_settings, monkeypatch):
    """Two RRF scores, highest first. Under RRF the value is
    ``Σ 1/(rank + k)``, so it is always positive and the top row wins."""
    edges = [_edge("u1", "first"), _edge("u2", "second")]
    res = MagicMock(edges=edges, edge_reranker_scores=[0.032, 0.016])
    g = _install(monkeypatch, res)

    out = await search_memory(base_settings, "q", limit=5, graphiti=g)

    assert out["success"] is True
    assert [r["reranker_score"] for r in out["results"]] == [0.032, 0.016]
    assert all(s > 0 for s in (r["reranker_score"] for r in out["results"]))
    assert out["diagnostics"]["scores_available"] is True


@pytest.mark.asyncio
async def test_score_is_matched_by_uuid_not_by_position(base_settings, monkeypatch):
    """Scores attached to the WRONG row is worse than no scores at all.

    The two lists are index-parallel upstream, so a naive ``zip`` passes here
    — the guard is the assertion that row i carries the score of ITS uuid,
    checked against a fixture where the two lists are deliberately reversed.
    """
    edges = [_edge("u1", "alpha"), _edge("u2", "beta")]
    res = MagicMock(edges=edges, edge_reranker_scores=[0.032, 0.016])
    g = _install(monkeypatch, res)

    out = await search_memory(base_settings, "q", limit=5, graphiti=g)

    by_uuid = {r["edge_uuid"]: r["reranker_score"] for r in out["results"]}
    assert by_uuid == {"u1": 0.032, "u2": 0.016}


@pytest.mark.asyncio
async def test_rows_carry_edge_uuid(base_settings, monkeypatch):
    edges = [_edge("u1", "alpha"), _edge("u2", "beta")]
    res = MagicMock(edges=edges, edge_reranker_scores=[0.032, 0.016])
    g = _install(monkeypatch, res)

    out = await search_memory(base_settings, "q", limit=5, graphiti=g)

    assert [r["edge_uuid"] for r in out["results"]] == ["u1", "u2"]


# ── AC3 / N4: vendor shape variations ─────────────────────────────────────


@pytest.mark.asyncio
async def test_bare_list_vendor_degrades_without_exception(base_settings, monkeypatch):
    """A bare list of edges has no score field at all — that is not an error."""
    edges = [_edge("u1", "alpha")]
    g = _install(monkeypatch, edges)

    out = await search_memory(base_settings, "q", limit=5, graphiti=g)

    assert out["success"] is True
    assert out["count"] == 1
    assert out["results"][0]["reranker_score"] is None
    assert out["diagnostics"]["scores_available"] is False


@pytest.mark.asyncio
async def test_empty_score_list_is_not_reported_as_zero(base_settings, monkeypatch):
    """N2: ``0.0`` is a legitimate-looking number. An empty list means
    "the reranker did not score this", which must not look like score 0."""
    edges = [_edge("u1", "alpha")]
    res = MagicMock(edges=edges, edge_reranker_scores=[])
    g = _install(monkeypatch, res)

    out = await search_memory(base_settings, "q", limit=5, graphiti=g)

    assert out["results"][0]["reranker_score"] is None
    assert out["diagnostics"]["scores_available"] is False


@pytest.mark.asyncio
async def test_short_score_list_leaves_tail_unscored_not_shifted(base_settings, monkeypatch):
    """N1: 2 edges but 1 score. ``zip`` would silently drop the second row's
    score onto the first row's neighbour; matching by uuid leaves it None."""
    edges = [_edge("u1", "alpha"), _edge("u2", "beta")]
    res = MagicMock(edges=edges, edge_reranker_scores=[0.032])
    g = _install(monkeypatch, res)

    out = await search_memory(base_settings, "q", limit=5, graphiti=g)

    by_uuid = {r["edge_uuid"]: r["reranker_score"] for r in out["results"]}
    assert by_uuid == {"u1": 0.032, "u2": None}


@pytest.mark.asyncio
async def test_identical_facts_with_distinct_uuids_keep_their_own_scores(
    base_settings, monkeypatch
):
    """N3: two edges can carry byte-identical text (measured 2026-10-03: 37
    facts vs 35 vectors). Matching on ``fact`` would collapse them."""
    edges = [_edge("u1", "same text"), _edge("u2", "same text")]
    res = MagicMock(edges=edges, edge_reranker_scores=[0.05, 0.02])
    g = _install(monkeypatch, res)

    out = await search_memory(base_settings, "q", limit=5, graphiti=g)

    by_uuid = {r["edge_uuid"]: r["reranker_score"] for r in out["results"]}
    assert by_uuid == {"u1": 0.05, "u2": 0.02}
    assert [r["fact"] for r in out["results"]] == ["same text", "same text"]


# ── AC4: the gate drops edges, scores must follow the survivors ────────────


@pytest.mark.asyncio
async def test_gate_dropping_edges_keeps_the_survivors_scores(base_settings, monkeypatch):
    """With the gate ON, edges are removed AFTER the score list was cut, so
    a positional zip drifts by exactly the number dropped."""
    from janus_graph.engine import search_memory as engine_module

    edges = [_edge("u1", "keep"), _edge("u2", "drop")]
    res = MagicMock(edges=edges, edge_reranker_scores=[0.05, 0.02])

    async def _fake(**kwargs):
        return res

    monkeypatch.setattr(engine_module, "graphiti_search", _fake)
    monkeypatch.setattr(engine_module, "resolve_cosine_gate_min", lambda cfg: 0.9)

    async def _gate(edges, *, query, gate_min, limit, graphiti, query_vector):
        # Mirrors the real signature: one list back, no dropped-count. The
        # facade derives `scores_dropped_by_gate` from the length delta, so a
        # gate that also returned a count would be a second source of truth.
        return [e for e in edges if e.uuid == "u1"]

    monkeypatch.setattr(engine_module, "_apply_cosine_gate", _gate)

    g = MagicMock()
    g.clients = MagicMock()
    g.clients.embedder.create = AsyncMock(return_value=None)

    out = await search_memory(base_settings, "q", limit=5, graphiti=g)

    assert [r["edge_uuid"] for r in out["results"]] == ["u1"]
    assert out["results"][0]["reranker_score"] == 0.05
    assert out["diagnostics"]["scores_dropped_by_gate"] == 1


@pytest.mark.asyncio
async def test_gate_off_reports_zero_dropped(base_settings, monkeypatch):
    """Production runs with the gate OFF by decision (measured 2026-10-05:
    every floor scored worse than OFF), so this is 0 in the real world."""
    from janus_graph.engine import search_memory as engine_module

    edges = [_edge("u1", "keep"), _edge("u2", "keep2")]
    res = MagicMock(edges=edges, edge_reranker_scores=[0.05, 0.02])

    async def _fake(**kwargs):
        return res

    monkeypatch.setattr(engine_module, "graphiti_search", _fake)
    monkeypatch.setattr(engine_module, "resolve_cosine_gate_min", lambda cfg: None)

    g = MagicMock()
    g.clients = MagicMock()
    g.clients.embedder.create = AsyncMock(return_value=None)

    out = await search_memory(base_settings, "q", limit=5, graphiti=g)

    assert out["diagnostics"]["scores_dropped_by_gate"] == 0


# ── AC5: score semantics must be named, never "relevance" ──────────────────


@pytest.mark.asyncio
async def test_score_semantics_is_rrf_when_reranker_is_rrf(base_settings, monkeypatch):
    """Explicitly opt in to rrf — the repo default in config.example.yaml is
    ``mmr``, so asserting "rrf" without setting it would test the fixture."""
    from janus_graph.engine import search_memory as engine_module

    base_settings.search.reranker = "rrf"
    edges = [_edge("u1", "a")]
    res = MagicMock(edges=edges, edge_reranker_scores=[0.03])
    g = _install(monkeypatch, res)

    out = await search_memory(base_settings, "q", limit=5, graphiti=g)

    assert out["diagnostics"]["reranker"] == "rrf"
    assert out["diagnostics"]["reranker"] == engine_module.normalize_reranker_name("rrf")
    assert out["diagnostics"]["score_semantics"] == "rrf_reciprocal_rank"


@pytest.mark.asyncio
async def test_score_semantics_tracks_a_non_rrf_reranker(base_settings, monkeypatch):
    from janus_graph.engine import search_memory as engine_module

    base_settings.search.reranker = "mmr"
    edges = [_edge("u1", "a")]
    res = MagicMock(edges=edges, edge_reranker_scores=[0.03])
    g = _install(monkeypatch, res)

    out = await search_memory(base_settings, "q", limit=5, graphiti=g)

    assert out["diagnostics"]["reranker"] == "mmr"
    assert out["diagnostics"]["score_semantics"] != "rrf_reciprocal_rank"


@pytest.mark.asyncio
async def test_no_row_uses_the_word_relevance(base_settings, monkeypatch):
    """The failure mode this guards: someone reads an RRF reciprocal-rank
    value as a cosine and compares it against ``sim_min_score``."""
    edges = [_edge("u1", "a")]
    res = MagicMock(edges=edges, edge_reranker_scores=[0.03])
    g = _install(monkeypatch, res)

    out = await search_memory(base_settings, "q", limit=5, graphiti=g)

    assert "relevance" not in out["results"][0]
    assert "score" not in out["results"][0]


# ── diagnostics bookkeeping ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_diagnostics_reports_candidate_and_return_counts(base_settings, monkeypatch):
    """``edge_candidates`` is what came back, ``edge_returned`` is what we
    emitted, and they must agree with ``count``.

    The facade does NOT truncate to ``limit`` itself — it sets
    ``search_config.limit`` (search_memory.py:275) and the vendor does the
    slicing, so a faithful fake has to honour the config it was handed.
    """
    from janus_graph.engine import search_memory as engine_module

    edges = [_edge(f"u{i}", f"f{i}") for i in range(4)]

    async def _fake(**kwargs):
        # Mirror the vendor: `return reranked_edges[:limit], edge_scores[:limit]`
        cap = kwargs["config"].limit
        res = MagicMock(edges=edges[:cap], edge_reranker_scores=[0.04, 0.03, 0.02, 0.01][:cap])
        return res

    monkeypatch.setattr(engine_module, "graphiti_search", _fake)
    monkeypatch.setattr(engine_module, "resolve_cosine_gate_min", lambda cfg: None)
    g = MagicMock()
    g.clients = MagicMock()
    g.clients.embedder.create = AsyncMock(return_value=None)

    out = await search_memory(base_settings, "q", limit=2, graphiti=g)

    d = out["diagnostics"]
    assert d["edge_candidates"] == 2, "what the vendor handed back"
    assert d["edge_returned"] == out["count"] == 2
    assert [r["reranker_score"] for r in out["results"]] == [0.04, 0.03]


@pytest.mark.asyncio
async def test_zero_results_still_has_diagnostics(base_settings, monkeypatch):
    res = MagicMock(edges=[], edge_reranker_scores=[])
    g = _install(monkeypatch, res)

    out = await search_memory(base_settings, "q", limit=5, graphiti=g)

    assert out["count"] == 0
    assert out["diagnostics"]["edge_candidates"] == 0
    assert out["diagnostics"]["scores_available"] is False


@pytest.mark.asyncio
async def test_old_contract_keys_are_untouched(base_settings, monkeypatch):
    edges = [_edge("u1", "a")]
    res = MagicMock(edges=edges, edge_reranker_scores=[0.03])
    g = _install(monkeypatch, res)

    out = await search_memory(base_settings, "q", limit=5, graphiti=g)

    assert {"success", "group_id", "query", "count", "results"} <= set(out)
    assert {"fact", "name", "valid_at", "invalid_at"} <= set(out["results"][0])


@pytest.mark.asyncio
async def test_error_path_does_not_gain_diagnostics(base_settings, monkeypatch):
    """A failure keeps its existing shape; adding a happy-path block must not
    leak into the error contract that the HTTP handler maps to status codes."""
    from janus_graph.engine import search_memory as engine_module

    async def _boom(**kwargs):
        raise ConnectionError("falkor down")

    monkeypatch.setattr(engine_module, "graphiti_search", _boom)
    g = MagicMock()

    out = await search_memory(base_settings, "q", limit=5, graphiti=g)

    assert out["success"] is False
    assert "diagnostics" not in out
