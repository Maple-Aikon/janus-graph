"""Tests for the 2026-10-04 /search/graph latency fix.

Three independent defects, three independent tests. Each fails if the
corresponding change is reverted, and none needs a live FalkorDB, a live
:8081, or the network.

  CAP      ``_attach_cosine`` must not hand more than ``embed_max_facts``
           facts to the embedding client. Measured 2026-10-04: :8081 runs
           ~24 ms/text, so 3219 rows (a 2-hop BFS) needed ~77 s and died on
           the embedding client's 10 s ClientTimeout, degrading every row.
  ORDER BY the BFS query must order before it limits. Measured: 73 -> 123
           distinct facts at the same cap of 200. Determinism is NOT
           asserted here: both shapes returned identical results on three
           consecutive runs, so ORDER BY bought distinctness, not
           reproducibility.
  CACHE    a warm per-text cache must remove texts from the upstream batch
           rather than re-send them, and must never cache a failure.

The double records the ``input`` list of every call that actually reaches
the transport, so a leak of cache hits back onto the wire is observable
rather than inferred.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from janus_graph.daemon.embedding_client import EmbeddingClient
from janus_graph.daemon.phase3_settings import DaemonSearchGraphSettings
from janus_graph.daemon.search_engine import _CYPHER_BFS, FactRow, SearchEngine


# ─── doubles ──────────────────────────────────────────────────────────


class StubEngineConfig:
    """Minimal stand-in for EngineConfig (SearchEngine only stores it)."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<stub EngineConfig>"


class _Resp:
    status = 200

    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return "error body"


class _Session:
    """Duck-typed aiohttp.ClientSession: only ``closed`` and ``post``."""

    def __init__(self, owner: "RecordingEmbedder"):
        self._owner = owner
        self.closed = False

    def post(self, url, json=None, headers=None):
        payload_in = list(json["input"])
        self._owner.batches.append(payload_in)
        return _Resp(self._owner.respond(payload_in))

    async def close(self):
        self.closed = True


class _ErrSession:
    """Like _Session but with an injectable post factory."""

    def __init__(self, owner, post):
        self._owner = owner
        self._post = post
        self.closed = False

    def post(self, url, json=None, headers=None):
        return self._post(url, json=json, headers=headers)

    async def close(self):
        self.closed = True


class RecordingEmbedder(EmbeddingClient):
    """Real EmbeddingClient (and its real cache) over a fake transport."""

    def __init__(
        self,
        settings: DaemonSearchGraphSettings,
        respond: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
    ):
        super().__init__(settings)
        self.batches: List[List[str]] = []
        self.respond = respond or self._default_respond
        # Pre-install the session so embed() never calls start() and never
        # opens a real aiohttp session (an unclosed one leaks a connector).
        self._session = _Session(self)  # type: ignore[assignment]

    @staticmethod
    def _default_respond(texts: List[str]) -> Dict[str, Any]:
        return {"data": [{"embedding": [float(len(t)), 1.0, 0.0]} for t in texts]}


def _rows(n: int) -> List[FactRow]:
    """FactRow needs all 6 required fields, not just ``fact``."""
    return [
        FactRow(
            fact="fact number %d" % i,
            source_entity="seed",
            target_entity="node%d" % i,
            edge_type="RELATES_TO",
            hop_distance=1,
            path=["seed", "node%d" % i],
        )
        for i in range(n)
    ]


def _engine(cap: int = 200) -> Tuple[SearchEngine, RecordingEmbedder, DaemonSearchGraphSettings]:
    settings = DaemonSearchGraphSettings(embed_max_facts=cap)
    embedder = RecordingEmbedder(settings)
    return SearchEngine(StubEngineConfig(), settings, embedder), embedder, settings


# ─── CAP ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_attach_cosine_trims_facts_before_embedding():
    """3219 rows must NOT become 3220 upstream texts.

    3220 texts at the measured ~24 ms/text is ~77 s, which is what killed
    the request before the 10 s ClientTimeout turned every cosine into None.
    """
    engine, embedder, settings = _engine(cap=200)

    await engine._attach_cosine(_rows(3219), "the query")

    assert len(embedder.batches) == 1, "one batched upstream call"
    assert len(embedder.batches[0]) == 201, "200 facts + 1 query"
    assert len(embedder.batches[0]) <= settings.embed_max_facts + 1


@pytest.mark.asyncio
async def test_attach_cosine_under_cap_is_untouched():
    """A batch already inside the cap must pass through unchanged."""
    engine, embedder, _ = _engine(cap=200)
    rows = _rows(5)

    await engine._attach_cosine(rows, "the query")

    assert embedder.batches == [["fact number %d" % i for i in range(5)] + ["the query"]]
    assert all(r.cosine is not None for r in rows)


@pytest.mark.asyncio
async def test_embed_cap_is_configurable_not_hardcoded():
    """Ops can retune the trim via settings, with no code edit."""
    engine, embedder, settings = _engine(cap=50)

    await engine._attach_cosine(_rows(500), "q")

    assert len(embedder.batches[0]) == settings.embed_max_facts + 1 == 51


@pytest.mark.asyncio
async def test_trim_does_not_mutate_the_caller_list():
    """``rows = rows[:cap]`` rebinds; it must not slice the caller's list."""
    engine, _, _ = _engine(cap=10)
    rows = _rows(50)

    scored = await engine._attach_cosine(rows, "q")

    assert len(rows) == 50, "caller keeps every row; only the embed batch is trimmed"
    assert scored == 10, "the return value says how many rows were actually scored"


@pytest.mark.asyncio
async def test_untrimmed_rows_do_not_force_degraded():
    """Drives the REAL ``SearchEngine.search()`` -- not a local re-derivation.

    The skipped tail must not be reported as an embedding failure. A
    trimming feature that sets ``degraded=True`` is indistinguishable from
    a broken embedder, so ``degraded`` is judged over the scored rows only.
    Asserted against the response the pipeline actually builds, otherwise
    this test would just be re-implementing the expression it checks.
    """
    from janus_graph.daemon.search_engine import SearchResponse

    engine, embedder, _ = _engine(cap=10)

    # search() runs the seed probe and a real redis connection first. Stub
    # both so the test exercises the REAL pipeline from Step 4 onward.
    engine._redis = object()

    async def _fake_probe(seeds):
        return list(seeds)

    engine._probe_seed_existence = _fake_probe

    # A 5-row BFS under the cap: every row must score, so nothing is skipped
    # and degraded must be False. This is the honest end-to-end assertion.
    async def _fake_bfs(req):
        return _rows(5)

    engine._cypher_bfs = _fake_bfs

    resp = await engine.search({"seed_entities": ["seed"], "max_hops": 2,
                                "min_cosine": 0.0, "mmr_lambda": 1.0, "limit": 5})
    assert isinstance(resp, SearchResponse)
    assert resp.degraded is False, "a fully scored batch must not be degraded"
    assert len(embedder.batches[0]) == 6, "5 facts + 1 query"

    # Now exceed the cap: the tail is skipped BY DESIGN and must not flip
    # the flag either -- the response should say so via a warning instead.
    async def _big_bfs(req):
        return _rows(50)

    engine._cypher_bfs = _big_bfs
    embedder.batches.clear()
    resp2 = await engine.search({"seed_entities": ["seed"], "max_hops": 2,
                                 "min_cosine": 0.0, "mmr_lambda": 1.0, "limit": 5})
    # Pass 1 already embedded facts 0-4 AND the query "seed" (6 texts).
    # Pass 2 reuses all six from cache and sends only the 5 new facts --
    # the query itself is cached too, which is why it is absent below.
    assert embedder.batches[0] == ["fact number %d" % i for i in range(5, 10)], \
        "only the uncached in-cap facts cross the wire; the query is cached"
    assert all(len(b) <= 11 for b in embedder.batches), \
        "no single upstream call may exceed the cap + query"
    assert embedder.cache_stats.hits == 6, "5 facts + the query were reused"
    assert any("embed_max_facts" in w for w in resp2.warnings), \
        "the skip must be disclosed as a warning, not hidden"
    assert resp2.count > 0, "the scored prefix must still return results"

    # THE decisive assertion, and the reason this test exists. Judging
    # degraded over ALL rows (not just the embedded prefix) makes the 40
    # deliberately-skipped rows read as 40 embedding failures -- turning
    # a working trim into a reported outage. Likewise `bool(warnings)`:
    # the cap warning is a disclosure, not a fault, so it must not set it.
    assert resp2.degraded is False, (
        "capping is a deliberate trim, not an embedder failure: "
        "degraded must stay False while the warning discloses the skip"
    )


@pytest.mark.asyncio
async def test_genuine_embed_failure_still_degrades():
    """The converse guard: a real embed failure MUST still set degraded.

    Without this, "degraded is always False" would pass every test above and
    the flag would be dead. A 500 from the endpoint is a real fault.
    """
    settings = DaemonSearchGraphSettings(embed_max_facts=10, embed_cache_size=100)

    class _Err:
        status = 500

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def json(self):
            return {}

        async def text(self):
            return "boom"

    class _Post:
        def __call__(self, url, json=None, headers=None):
            return _Err()

    embedder = RecordingEmbedder(settings)
    embedder._session = _ErrSession(embedder, _Post())
    engine = SearchEngine(StubEngineConfig(), settings, embedder)
    engine._redis = object()

    async def _probe(seeds):
        return list(seeds)

    engine._probe_seed_existence = _probe

    async def _bfs(req):
        return _rows(5)

    engine._cypher_bfs = _bfs

    resp = await engine.search({"seed_entities": ["seed"], "max_hops": 2,
                                "min_cosine": 0.0, "mmr_lambda": 1.0, "limit": 5})
    assert resp.degraded is True, "a 500 from the embedder IS a degradation"
    assert resp.count == 0, "no row can be scored without a cosine"


# ─── ORDER BY ────────────────────────────────────────────────────────


def test_bfs_query_orders_before_it_limits():
    """A LIMIT with no ORDER BY slices an unordered set."""
    assert "ORDER BY" in _CYPHER_BFS
    assert _CYPHER_BFS.index("ORDER BY") < _CYPHER_BFS.index("LIMIT {cap}")
    assert "hop_distance ASC" in _CYPHER_BFS
    assert "edge_fact ASC" in _CYPHER_BFS


# ─── CACHE ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_second_call_sends_nothing_upstream():
    """A fully warm cache must not open a request at all."""
    embedder = RecordingEmbedder(DaemonSearchGraphSettings(embed_cache_size=100))

    first = await embedder.embed(["alpha", "beta"])
    assert all(e.embedding is not None for e in first)
    assert embedder.batches == [["alpha", "beta"]], "cold call fetches both"

    embedder.batches.clear()
    second = await embedder.embed(["alpha", "beta"])

    assert [e.embedding for e in second] == [e.embedding for e in first]
    assert embedder.batches == [], "warm cache must not reach the transport"
    assert embedder.cache_stats.hits == 2


@pytest.mark.asyncio
async def test_partial_cache_sends_only_misses():
    """A half-warm cache puts ONLY the uncached text on the wire.

    The repeated 'beta' is a cache hit and must not be re-sent -- resending
    is exactly what would keep the 10 s timeout alive.
    """
    embedder = RecordingEmbedder(DaemonSearchGraphSettings(embed_cache_size=100))
    await embedder.embed(["alpha", "beta", "gamma"])
    embedder.batches.clear()

    out = await embedder.embed(["alpha", "beta", "beta", "delta"])

    assert embedder.batches == [["delta"]]
    assert len(out) == 4
    assert all(e.embedding is not None for e in out)
    # Positional integrity: each text keeps its OWN vector.
    vectors = {e.text: e.embedding for e in out}
    assert vectors["alpha"] == [5.0, 1.0, 0.0]
    assert vectors["beta"] == [4.0, 1.0, 0.0]
    assert vectors["delta"] == [5.0, 1.0, 0.0]


@pytest.mark.asyncio
async def test_failures_are_not_cached():
    """A failed embed must retry next time, not cache a None forever.

    NOTE on why this is asserted via the batch list and not via the cache:
    ``LRUCacheBackend.get()`` returns ``None`` both for "absent" and for
    "present but None", so a cached None is indistinguishable from a miss
    and the failure would heal itself anyway. The observable contract that
    actually matters is that the retry reaches the transport.
    """

    def _empty(texts: List[str]) -> Dict[str, Any]:
        return {"data": []}

    embedder = RecordingEmbedder(
        DaemonSearchGraphSettings(embed_cache_size=100), respond=_empty
    )
    first = await embedder.embed(["flaky"])
    assert first[0].embedding is None

    embedder.respond = RecordingEmbedder._default_respond
    embedder.batches.clear()
    second = await embedder.embed(["flaky"])

    assert second[0].embedding == [5.0, 1.0, 0.0]
    assert embedder.batches == [["flaky"]], "a failed text must be retried upstream"


@pytest.mark.asyncio
async def test_http_error_keeps_cache_hits_intact():
    """A 500 on the misses must not wipe the rows already cached."""
    settings = DaemonSearchGraphSettings(embed_cache_size=100)
    embedder = RecordingEmbedder(settings)
    await embedder.embed(["alpha"])
    embedder.batches.clear()

    class _Err:
        status = 500

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def json(self):
            return {}

        async def text(self):
            return "boom"

    class _ErrPost:
        def __call__(self, url, json=None, headers=None):
            embedder.batches.append(list(json["input"]))
            return _Err()

    embedder._session = _ErrSession(embedder, _ErrPost())
    out = await embedder.embed(["alpha", "beta"])

    assert embedder.batches == [["beta"]], "the cached text stays off the wire"
    assert out[0].embedding is not None, "alpha survives via cache"
    assert out[1].embedding is None, "beta failed upstream"


@pytest.mark.asyncio
async def test_lru_eviction_respects_max_size():
    """The cache honours its configured bound instead of growing forever."""
    embedder = RecordingEmbedder(DaemonSearchGraphSettings(embed_cache_size=2))
    await embedder.embed(["a", "b", "c"])       # 3 texts, capacity 2
    assert embedder._cache.size() == 2

    embedder.batches.clear()
    await embedder.embed(["a", "b", "c"])
    assert embedder.batches == [["a"]], "only the evicted entry is re-fetched"
