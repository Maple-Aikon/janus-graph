"""Unit tests for ``SearchEngine._probe_seed_existence``.

Why this file exists
--------------------
The seed probe answers "does this seed exist?" by running one existence
query per seed. It used to catch *every* exception and treat the seed as
absent, so a transport failure (FalkorDB down, socket timeout) and a parse
rejection were both indistinguishable from "that entity is not in the KG".

The caller (``search()``) turns an empty list into
``SearchBackendError("SEED_NOT_FOUND")`` -> HTTP 404. So a dead database was
reported to callers as "you asked for an entity that does not exist", and
the caller was told to fix their query. The log line that recorded the real
cause was emitted at DEBUG, i.e. invisible in normal operation.

The sibling method ``_cypher_bfs`` already does the right thing: it re-raises
through ``_classify_cypher_error``. These tests pin the probe to the same
discipline.

Coverage
--------
- a transport error must surface as FALKOR_DOWN, never as "seed absent"
- a FalkorDB parse rejection must surface as INVALID_CYPHER, never as
  "seed absent" (an INVALID_CYPHER here means our own generated query is
  malformed -- a server-side defect, so it must not read as an outage)
- a genuine miss (FalkorDB answers, zero rows) must STILL return an empty
  list, so the existing 404 SEED_NOT_FOUND path is unchanged
- a partial match (some seeds found, some absent) returns only the found ones

No live FalkorDB, no network, no production code changes.
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from janus_graph.daemon.phase3_settings import DaemonSearchGraphSettings
from janus_graph.daemon.search_engine import SearchBackendError, SearchEngine


def _engine() -> SearchEngine:
    """A SearchEngine with a mocked embedding client and real settings."""
    return SearchEngine(
        engine_config=MagicMock(),
        search_settings=DaemonSearchGraphSettings(),
        embedding_client=MagicMock(),
    )


def _exists_result(found: bool) -> List[Any]:
    """Wrap a probe reply the way FalkorDB ``--compact`` does.

    ``MATCH (n {name: $name}) RETURN n.name LIMIT 1`` has one column, so the
    header is a single cell and ``rows`` is empty or one row long.
    """
    header = [[1, "n.name"]]
    rows = [[2, "Seed"]] if found else []
    return [header, rows, ["Cached execution: 0"]]


def _attach(engine: SearchEngine, side_effect: Any = None, result: Any = None):
    """Attach a mocked redis: either raising ``side_effect`` or returning
    ``result``.
    """
    if side_effect is not None:
        redis = AsyncMock(side_effect=side_effect)
    else:
        redis = AsyncMock(return_value=result)
    engine._redis = MagicMock()
    engine._redis.execute_command = redis
    return redis


# ── transport failure must not masquerade as "seed absent" ──────────────


@pytest.mark.asyncio
async def test_connection_error_surfaces_as_falkor_down():
    """A refused connection is a dead database, not a missing entity."""
    engine = _engine()
    _attach(engine, side_effect=ConnectionError("connection refused"))

    with pytest.raises(SearchBackendError) as exc:
        await engine._probe_seed_existence(["Alpha"])

    assert exc.value.code == "FALKOR_DOWN"
    assert "connection refused" in exc.value.message


@pytest.mark.asyncio
async def test_timeout_surfaces_as_falkor_down():
    """A socket timeout on the probe is likewise an outage."""
    engine = _engine()
    _attach(engine, side_effect=TimeoutError("timed out"))

    with pytest.raises(SearchBackendError) as exc:
        await engine._probe_seed_existence(["Alpha"])

    assert exc.value.code == "FALKOR_DOWN"


# ── parse rejection must not masquerade as an outage, nor as a miss ──────


@pytest.mark.asyncio
async def test_falkordb_parse_rejection_surfaces_as_invalid_cypher():
    """"Invalid input P" is our bug, not an outage (see _classify_cypher_error)."""
    engine = _engine()
    _attach(engine, side_effect=RuntimeError("Invalid input P: expected ..."))

    with pytest.raises(SearchBackendError) as exc:
        await engine._probe_seed_existence(["Alpha"])

    assert exc.value.code == "INVALID_CYPHER"


# ── genuine misses must keep their existing behaviour ────────────────────


@pytest.mark.asyncio
async def test_genuine_miss_returns_empty_list():
    """FalkorDB answered and found nothing -> empty list, no exception.

    This is what makes the caller's 404 SEED_NOT_FOUND correct; it must not
    change when the error paths above start raising.
    """
    engine = _engine()
    _attach(engine, result=_exists_result(found=False))

    assert await engine._probe_seed_existence(["Ghost"]) == []


@pytest.mark.asyncio
async def test_hit_returns_the_seed_name():
    engine = _engine()
    _attach(engine, result=_exists_result(found=True))

    assert await engine._probe_seed_existence(["Alpha"]) == ["Alpha"]


@pytest.mark.asyncio
async def test_partial_match_returns_only_found_seeds():
    engine = _engine()
    redis = AsyncMock(
        side_effect=[_exists_result(True), _exists_result(False)]
    )
    engine._redis = MagicMock()
    engine._redis.execute_command = redis

    assert await engine._probe_seed_existence(["Alpha", "Ghost"]) == ["Alpha"]
