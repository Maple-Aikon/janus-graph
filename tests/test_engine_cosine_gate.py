# ===========================================================================
# Post-hoc TRUE-COSINE relevance gate (2026-10-01)
# ===========================================================================
#
# The gate exists because three cheaper fixes were measured and killed:
#   1. MMR reranker scores CANNOT gate. MMR is
#      ``lambda*cos + (lambda-1)*max_sim`` — a relevance+redundancy
#      mixture. Measured on-topic min 0.3191 < off-topic max 0.3281, so
#      the distributions overlap by construction.
#   2. The FalkorDB driver score is ``(1+cos)/2``, NOT cosine. A floor of
#      F against it really means ``cos >= 2F - 1``, so "raise
#      sim_min_score to 0.92" was calibrating against the wrong scale.
#   3. Raising ``sim_min_score`` only guards the cosine arm; graphiti_core
#      passes NO min_score to the bm25/fulltext arm at all, so lexical
#      junk keeps returning edges no matter how high sim_min_score goes.
#
# These tests pin the properties the gate MUST hold, so a future refactor
# cannot quietly regress it back into an MMR-based or driver-score-based
# filter. They are unit tests with fake embeddings — no FalkorDB needed.

from unittest.mock import AsyncMock, MagicMock

import pytest

from janus_graph.engine import search_memory as engine_module
from janus_graph.engine.search_memory import search_memory


@pytest.fixture
def base_settings():
    """Fresh JanusSettings per test.

    Defined locally rather than imported: the gate tests mutate
    ``cfg.search.cosine_gate_min``, and sharing the other test module's
    fixture would let those mutations leak across files.
    """
    from janus_graph.config import JanusSettings

    return JanusSettings()


def _edge(uuid, fact="a fact", vec=None):
    """A minimal stand-in for graphiti_core EntityEdge.

    Only ``uuid`` and ``fact`` are read by the engine facade; ``vec`` is
    used by the test to know which embedding the fake loader will return.
    """
    e = MagicMock()
    e.uuid = uuid
    e.fact = fact
    e.name = uuid
    e.valid_at = "2026-01-01"
    e.invalid_at = ""
    e._test_vec = vec
    return e


class _FakeEmbedder:
    """Records embed calls; returns a fixed vector."""

    def __init__(self, vec=(1.0, 0.0, 0.0)):
        self.vec = list(vec)
        self.calls = []

    async def create(self, input_data=None, **kwargs):
        self.calls.append(input_data)
        return list(self.vec)


def _patch_gate_deps(monkeypatch, edges_by_uuid):
    """Patch ``get_embeddings_for_edges`` to serve ``edges_by_uuid``.

    Returns a dict with a ``calls`` list recording how many edges each call
    was asked for, so tests can assert the gate only loads what it must.
    """
    calls = []

    async def fake_get_embeddings(driver, edges):
        calls.append(len(edges))
        return {e.uuid: edges_by_uuid[e.uuid] for e in edges if e.uuid in edges_by_uuid}

    monkeypatch.setattr(
        engine_module, "get_embeddings_for_edges", fake_get_embeddings, raising=True
    )
    return {"calls": calls}


def _graphiti(embedder_vec=(1.0, 0.0, 0.0)):
    g = MagicMock()
    g.clients = MagicMock()
    g.clients.embedder = _FakeEmbedder(embedder_vec)
    g.clients.driver = MagicMock()
    return g


@pytest.mark.asyncio
async def test_cosine_gate_disabled_by_default(base_settings, monkeypatch):
    """Default config MUST leave search_memory byte-for-byte unchanged.

    Guards the promise that adopting this gate is opt-in: with no config
    key set, no embedding call, no embedding load, no filtering.
    """
    assert base_settings.search.cosine_gate_min is None

    g = _graphiti()
    embeds = _patch_gate_deps(monkeypatch, {})

    mock_search = AsyncMock(return_value=MagicMock(edges=[_edge("a"), _edge("b")]))
    original = engine_module.graphiti_search
    engine_module.graphiti_search = mock_search
    try:
        result = await search_memory(base_settings, "q", limit=5, graphiti=g)
    finally:
        engine_module.graphiti_search = original

    assert result["success"] is True
    assert result["count"] == 2
    assert g.clients.embedder.calls == [], "gate off must not embed the query"
    assert embeds["calls"] == [], "gate off must not load edge embeddings"
    # limit passed through unchanged, no over-fetch
    assert mock_search.await_args.kwargs["config"].limit == 5


@pytest.mark.asyncio
async def test_cosine_gate_drops_below_floor(base_settings, monkeypatch):
    """Edges below the cosine floor are dropped; edges above survive.

    Query vector is (1,0,0). An edge aligned with it has cos 1.0; an edge
    opposite has cos -1.0. Floor 0.9 must keep the first and drop the
    second. This pins TRUE cosine semantics — an implementation that
    compared against (1+cos)/2 or against MMR would fail here.
    """
    base_settings.search.cosine_gate_min = 0.9
    g = _graphiti(embedder_vec=(1.0, 0.0, 0.0))

    edges = [_edge("keep", vec=[1.0, 0.0, 0.0]), _edge("drop", vec=[-1.0, 0.0, 0.0])]
    _patch_gate_deps(monkeypatch, {"keep": [1.0, 0.0, 0.0], "drop": [-1.0, 0.0, 0.0]})

    mock_search = AsyncMock(return_value=MagicMock(edges=edges))
    original = engine_module.graphiti_search
    engine_module.graphiti_search = mock_search
    try:
        result = await search_memory(base_settings, "q", limit=5, graphiti=g)
    finally:
        engine_module.graphiti_search = original

    assert result["success"] is True
    assert result["count"] == 1
    assert result["results"][0]["name"] == "keep"


@pytest.mark.asyncio
async def test_cosine_gate_covers_bm25_arm(base_settings, monkeypatch):
    """The gate must filter edges that arrived with NO cosine-arm score.

    graphiti_core's bm25/fulltext arm passes no ``min_score`` to the
    driver, so a lexical hit can surface an edge whose only ranking signal
    was a text match. In this test the fake search returns edges and NO
    ``edge_reranker_scores`` at all — exactly the shape of a score-less
    bm25 hit. A gate built on reranker scores would have nothing to read
    and could not drop them; a gate on true cosine can.
    """
    base_settings.search.cosine_gate_min = 0.5
    g = _graphiti(embedder_vec=(1.0, 0.0, 0.0))

    # SearchResults without edge_reranker_scores populated at all.
    bm25_shape = MagicMock(spec=["edges"])
    bm25_shape.edges = [
        _edge("lexical_hit_but_offtopic", vec=[0.0, 1.0, 0.0]),
        _edge("lexical_hit_and_relevant", vec=[1.0, 0.0, 0.0]),
    ]
    _patch_gate_deps(
        monkeypatch,
        {
            "lexical_hit_but_offtopic": [0.0, 1.0, 0.0],
            "lexical_hit_and_relevant": [1.0, 0.0, 0.0],
        },
    )

    mock_search = AsyncMock(return_value=bm25_shape)
    original = engine_module.graphiti_search
    engine_module.graphiti_search = mock_search
    try:
        result = await search_memory(base_settings, "q", limit=5, graphiti=g)
    finally:
        engine_module.graphiti_search = original

    assert result["success"] is True
    assert result["count"] == 1
    assert result["results"][0]["name"] == "lexical_hit_and_relevant"


@pytest.mark.asyncio
async def test_cosine_gate_uses_true_cosine_not_driver_score(base_settings, monkeypatch):
    """Floor must be interpreted as COSINE, not as FalkorDB's (1+cos)/2.

    With query (1,0,0) and edge (0.6, 0.8, 0.0), true cosine is 0.6 but the
    driver score would be 0.8. A floor of 0.7 therefore keeps the edge in
    cosine units and drops it in driver-score units. This is the exact
    scale confusion that produced the wrong "0.92" recommendation.
    """
    base_settings.search.cosine_gate_min = 0.7
    g = _graphiti(embedder_vec=(1.0, 0.0, 0.0))

    edges = [_edge("borderline", vec=[0.6, 0.8, 0.0])]
    _patch_gate_deps(monkeypatch, {"borderline": [0.6, 0.8, 0.0]})

    mock_search = AsyncMock(return_value=MagicMock(edges=edges))
    original = engine_module.graphiti_search
    engine_module.graphiti_search = mock_search
    try:
        result = await search_memory(base_settings, "q", limit=5, graphiti=g)
    finally:
        engine_module.graphiti_search = original

    # cos = 0.6 < 0.7 -> dropped, even though driver score (1+0.6)/2 = 0.8
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_cosine_gate_overfetches_and_truncates_back(base_settings, monkeypatch):
    """Gate must over-fetch, then restore the caller's limit.

    Measured on this host: a true match can sit at rank 11 when limit=10,
    so gating without over-fetch silently loses real results. The gate
    must therefore ask graphiti for ``limit * overfetch`` and hand back at
    most ``limit``.
    """
    base_settings.search.cosine_gate_min = 0.9
    base_settings.search.cosine_gate_overfetch = 2
    g = _graphiti(embedder_vec=(1.0, 0.0, 0.0))

    # 4 edges: ranks 1-2 irrelevant, ranks 3-4 relevant. limit=2 -> fetch 4.
    edges = [
        _edge("off1", vec=[0.0, 1.0, 0.0]),
        _edge("off2", vec=[0.0, 1.0, 0.0]),
        _edge("on1", vec=[1.0, 0.0, 0.0]),
        _edge("on2", vec=[1.0, 0.0, 0.0]),
    ]
    _patch_gate_deps(monkeypatch, {e.uuid: e._test_vec for e in edges})

    mock_search = AsyncMock(return_value=MagicMock(edges=edges))
    original = engine_module.graphiti_search
    engine_module.graphiti_search = mock_search
    try:
        result = await search_memory(base_settings, "q", limit=2, graphiti=g)
    finally:
        engine_module.graphiti_search = original

    assert mock_search.await_args.kwargs["config"].limit == 4, "must over-fetch"
    assert result["count"] == 2, "must truncate back to the caller's limit"
    assert [r["name"] for r in result["results"]] == ["on1", "on2"]


@pytest.mark.asyncio
async def test_cosine_gate_drops_edges_missing_embedding(base_settings, monkeypatch):
    """Unscorable edges are dropped and counted, not silently kept.

    Keeping them would defeat the gate precisely for the malformed rows it
    exists to suppress; dropping them silently would hide a data problem.
    The test pins the drop; the log line is the visibility.
    """
    base_settings.search.cosine_gate_min = 0.9
    g = _graphiti(embedder_vec=(1.0, 0.0, 0.0))

    edges = [_edge("no_embedding", vec=None), _edge("good", vec=[1.0, 0.0, 0.0])]
    _patch_gate_deps(monkeypatch, {"good": [1.0, 0.0, 0.0]})  # no entry for no_embedding

    mock_search = AsyncMock(return_value=MagicMock(edges=edges))
    original = engine_module.graphiti_search
    engine_module.graphiti_search = mock_search
    try:
        result = await search_memory(base_settings, "q", limit=5, graphiti=g)
    finally:
        engine_module.graphiti_search = original

    assert result["count"] == 1
    assert result["results"][0]["name"] == "good"


@pytest.mark.asyncio
async def test_cosine_gate_normalises_multiline_query(base_settings, monkeypatch):
    """Gate and search must score against the SAME query vector.

    graphiti_core normalises newlines to spaces before embedding, but only
    when it embeds internally. Since the facade embeds once and passes the
    vector through, it owns that normalisation. If it did not, a
    multi-line query would be gated against a different vector than the
    one search used.
    """
    base_settings.search.cosine_gate_min = 0.9
    g = _graphiti(embedder_vec=(1.0, 0.0, 0.0))

    edges = [_edge("keep", vec=[1.0, 0.0, 0.0])]
    _patch_gate_deps(monkeypatch, {"keep": [1.0, 0.0, 0.0]})

    mock_search = AsyncMock(return_value=MagicMock(edges=edges))
    original = engine_module.graphiti_search
    engine_module.graphiti_search = mock_search
    try:
        result = await search_memory(
            base_settings, "line one" + chr(10) + "line two", limit=5, graphiti=g
        )
    finally:
        engine_module.graphiti_search = original

    assert g.clients.embedder.calls == [["line one line two"]]
    passed = mock_search.await_args.kwargs["query_vector"]
    assert passed == [1.0, 0.0, 0.0], "the gate and search must share one vector"
    assert result["count"] == 1


@pytest.mark.asyncio
async def test_cosine_gate_fails_open_on_embedding_error(base_settings, monkeypatch):
    """A broken embedding load must not turn into an empty search.

    The gate is a relevance filter, not a correctness dependency. If it
    cannot load embeddings it returns the un-gated edges and logs — it
    must never raise into the facade's error contract, which would turn a
    partial degradation into ``success: False``.
    """
    base_settings.search.cosine_gate_min = 0.9
    g = _graphiti(embedder_vec=(1.0, 0.0, 0.0))

    edges = [_edge("a"), _edge("b")]

    async def boom(driver, es):
        raise RuntimeError("falkordb dropped the connection")

    monkeypatch.setattr(engine_module, "get_embeddings_for_edges", boom, raising=True)

    mock_search = AsyncMock(return_value=MagicMock(edges=edges))
    original = engine_module.graphiti_search
    engine_module.graphiti_search = mock_search
    try:
        result = await search_memory(base_settings, "q", limit=5, graphiti=g)
    finally:
        engine_module.graphiti_search = original

    assert result["success"] is True, "must fail open, not error out"
    assert result["count"] == 2


@pytest.mark.asyncio
async def test_cosine_gate_all_edges_dropped_is_success_zero(base_settings, monkeypatch):
    """A fully-gated query returns success=True, count=0, results=[].

    The HTTP handler returns 200 + count/results for any success=True, and
    ``count == 0`` is an established valid outcome (see the existing
    facade smoke test). The gate relies on that, so pin it.
    """
    base_settings.search.cosine_gate_min = 0.9
    g = _graphiti(embedder_vec=(1.0, 0.0, 0.0))

    edges = [_edge("junk1", vec=[0.0, 1.0, 0.0]), _edge("junk2", vec=[0.0, 0.0, 1.0])]
    _patch_gate_deps(monkeypatch, {"junk1": [0.0, 1.0, 0.0], "junk2": [0.0, 0.0, 1.0]})

    mock_search = AsyncMock(return_value=MagicMock(edges=edges))
    original = engine_module.graphiti_search
    engine_module.graphiti_search = mock_search
    try:
        result = await search_memory(base_settings, "zzzz qqq", limit=5, graphiti=g)
    finally:
        engine_module.graphiti_search = original

    assert result["success"] is True
    assert result["count"] == 0
    assert result["results"] == []


# --- config resolver behaviour -------------------------------------------


def test_cosine_gate_min_defaults_to_none():
    """Disabled by default so adopting the gate is opt-in."""
    from janus_graph.config import JanusSettings, resolve_cosine_gate_min

    assert resolve_cosine_gate_min(JanusSettings()) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.845", 0.845),
        ("off", None),
        ("none", None),
        ("disabled", None),
        ("", None),
        ("abc", None),  # non-numeric -> fail soft to default None
        ("9.9", None),  # out of range -> fail soft
        ("-9.9", None),  # out of range -> fail soft
    ],
)
def test_cosine_gate_min_env_fail_soft(monkeypatch, raw, expected):
    """Bad env values must never enable a broken gate or brick search."""
    from janus_graph.config import JanusSettings, resolve_cosine_gate_min

    monkeypatch.setenv("JANUS_SEARCH__COSINE_GATE_MIN", raw)
    assert resolve_cosine_gate_min(JanusSettings()) == expected


def test_cosine_gate_overfetch_fail_soft(monkeypatch):
    """Overfetch must stay within [1, 10]; bad env falls back to 2."""
    from janus_graph.config import JanusSettings, resolve_cosine_gate_overfetch

    cfg = JanusSettings()
    monkeypatch.delenv("JANUS_SEARCH__COSINE_GATE_OVERFETCH", raising=False)
    assert resolve_cosine_gate_overfetch(cfg) == 2

    monkeypatch.setenv("JANUS_SEARCH__COSINE_GATE_OVERFETCH", "0")
    assert resolve_cosine_gate_overfetch(cfg) == 2
    monkeypatch.setenv("JANUS_SEARCH__COSINE_GATE_OVERFETCH", "99")
    assert resolve_cosine_gate_overfetch(cfg) == 2
    monkeypatch.setenv("JANUS_SEARCH__COSINE_GATE_OVERFETCH", "x")
    assert resolve_cosine_gate_overfetch(cfg) == 2
    monkeypatch.setenv("JANUS_SEARCH__COSINE_GATE_OVERFETCH", "3")
    assert resolve_cosine_gate_overfetch(cfg) == 3


def test_legacy_search_defaults_unchanged():
    """The gate must not have moved any pre-existing default.

    ``sim_min_score`` 0.6 in particular is pinned by other tests against
    graphiti_core's DEFAULT_MIN_SCORE; changing it would alter behaviour
    for every existing deployment.
    """
    from janus_graph.config import JanusSettings

    cfg = JanusSettings()
    assert cfg.search.sim_min_score == 0.6
    assert cfg.search.mmr_lambda == 0.5
    assert cfg.search.reranker_min_score == -2.0
    assert cfg.search.cosine_gate_overfetch == 2
