"""Unit tests for ``SearchEngine._cypher_bfs``.

Untested private method (previously only reachable through ``search()``).
It builds the Cypher BFS text, executes it against FalkorDB, and parses the
``--compact`` reply into ``FactRow`` objects.

Coverage:
  QUERY BUILD
  - seed literals are inlined as single-quoted strings
  - internal single quotes in a seed are backslash-escaped (FalkorDB does NOT
    use SQL-style quote doubling)
  - ``{max_hops}`` and ``{cap}`` placeholders are substituted from the
    request and from ``settings.max_facts_per_query``
  - the query is sent as GRAPH.QUERY / graphiti_memory / <cypher> / --compact
  BACKEND ERROR
  - any exception from execute_command becomes SearchBackendError("FALKOR_DOWN")
  ROW PARSING (all via a mocked redis, no live Falkor)
  - happy path: 6 flat columns -> FactRow
  - multi-line summary collapses to its first non-blank, stripped line
  - blank/whitespace-only summary is skipped
  - non-str summary is skipped
  - missing edge_type falls back to "RELATES_TO"
  - hop_distance None defaults to 1
  - source/target fall back to path_nodes[0] when the seed column is null
  - a short row (< 6 cols) raises inside the try and is skipped, not fatal
  - a row whose hop is non-numeric is skipped without killing the batch
  - one bad row does not discard the good rows in the same batch
  - empty result / non-list result -> empty list

No live FalkorDB, no network, no production code changes.

NOTE: multi-line fixtures are built with ``chr(10)`` rather than escape
sequences so this file stays free of backslash literals.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from janus_graph.daemon.phase3_settings import DaemonSearchGraphSettings
from janus_graph.daemon.search_engine import SearchBackendError, SearchEngine

_NL = chr(10)

#: First two lines blank/whitespace, third carries the fact. The parser must
#: skip blanks and strip the result.
MULTILINE_SUMMARY = _NL + "   " + _NL + "  First meaningful line  " + _NL + "second line" + _NL

#: Empty or whitespace-only summaries -> the row is dropped.
BLANK_SUMMARIES = [
    "",
    "   ",
    _NL,
    "  " + _NL + " " + _NL + " ",
]


# ─── helpers ─────────────────────────────────────────────────────────────


def _engine(*, cap: int = 5000) -> SearchEngine:
    """A SearchEngine with a mocked embedding client and real settings."""
    return SearchEngine(
        engine_config=MagicMock(),
        search_settings=DaemonSearchGraphSettings(max_facts_per_query=cap),
        embedding_client=MagicMock(),
    )


def _compact_result(rows: List[List[Any]]) -> List[Any]:
    """Wrap raw rows the way FalkorDB ``--compact`` does.

    Shape is ``[header, rows, stats]``, each cell a ``[type_code, value]`` pair.
    """
    cells = [[1, f"col{i}"] for i in range(6)]
    return [cells, rows, ["Cached execution: 0"]]


def _req(seeds, max_hops: int = 2) -> Dict[str, Any]:
    return {"seeds": list(seeds), "max_hops": max_hops}


def _attach_redis(engine: SearchEngine, result) -> AsyncMock:
    """Give the engine a mocked redis whose GRAPH.QUERY returns ``result``."""
    redis = AsyncMock(return_value=result)
    engine._redis = MagicMock()
    engine._redis.execute_command = redis
    return redis


async def _run(engine: SearchEngine, req, redis_result) -> List[Any]:
    """Attach a mocked redis and run ``_cypher_bfs``."""
    _attach_redis(engine, redis_result)
    return await engine._cypher_bfs(req)


# ─── query build ────────────────────────────────────────────────────────


async def test_seed_literals_are_inlined_quoted():
    engine = _engine()
    redis = _attach_redis(engine, _compact_result([]))

    await engine._cypher_bfs(_req(["Alpha", "Beta"]))

    assert "UNWIND ['Alpha', 'Beta'] AS seed_name" in redis.await_args.args[2]


async def test_single_quote_in_seed_is_backslash_escaped():
    """A seed containing an apostrophe is backslash-escaped.

    Was previously asserted as quote-doubling (``'O''Brien'``). That is
    SQL/openCypher convention but is a PARSE ERROR in FalkorDB v8.9.x, which
    uses backslash escaping inside string literals. Corrected 2026-10-01.
    """
    engine = _engine()
    redis = _attach_redis(engine, _compact_result([]))

    await engine._cypher_bfs(_req(["O'Brien"]))

    assert "UNWIND ['O" + chr(92) + chr(39) + "Brien'] AS seed_name" in redis.await_args.args[2]


async def test_max_hops_placeholder_is_substituted():
    engine = _engine()
    redis = _attach_redis(engine, _compact_result([]))

    await engine._cypher_bfs(_req(["A"], max_hops=4))

    cypher = redis.await_args.args[2]
    assert "[*1..4]" in cypher
    assert "{max_hops}" not in cypher


async def test_cap_placeholder_uses_max_facts_per_query():
    """The LIMIT comes from settings, and is int()-coerced."""
    engine = _engine(cap=77)
    redis = _attach_redis(engine, _compact_result([]))

    await engine._cypher_bfs(_req(["A"]))

    cypher = redis.await_args.args[2]
    assert "LIMIT 77" in cypher
    assert "{cap}" not in cypher


async def test_query_is_sent_with_graph_query_and_compact_flag():
    engine = _engine()
    redis = _attach_redis(engine, _compact_result([]))

    await engine._cypher_bfs(_req(["A"]))

    args = redis.await_args.args
    assert args[0] == "GRAPH.QUERY"
    assert args[1] == "graphiti_memory"
    assert args[3] == "--compact"


# ─── backend failure ────────────────────────────────────────────────────


async def test_execute_error_becomes_falkor_down():
    engine = _engine()
    engine._redis = MagicMock()
    engine._redis.execute_command = AsyncMock(side_effect=ConnectionError("connection refused"))

    with pytest.raises(SearchBackendError) as exc:
        await engine._cypher_bfs(_req(["A"]))

    assert exc.value.code == "FALKOR_DOWN"
    assert "connection refused" in exc.value.message


# ─── row parsing ────────────────────────────────────────────────────────


async def test_happy_path_builds_fact_row():
    """A well-formed 6-column row becomes one FactRow."""
    engine = _engine()
    result = _compact_result(
        [
            [
                [2, "MENTIONS"],  # edge_type
                [3, 2],  # hop_distance
                [6, ["Seed", "Mid", "Target"]],  # path_nodes
                [2, "Fact text here"],  # first_summary
                [2, "Seed"],  # seed_in_path
                [2, "Target"],  # target_in_path
            ]
        ]
    )

    rows = await _run(engine, _req(["Seed"]), result)

    assert len(rows) == 1
    row = rows[0]
    assert row.fact == "Fact text here"
    assert row.edge_type == "MENTIONS"
    assert row.hop_distance == 2
    assert row.path == ["Seed", "Mid", "Target"]
    assert row.source_entity == "Seed"
    assert row.target_entity == "Target"
    assert row.invalid_at is None
    assert row.cosine is None


async def test_multiline_summary_collapses_to_first_nonblank_line():
    engine = _engine()
    result = _compact_result(
        [
            [
                [2, "MENTIONS"],
                [3, 1],
                [6, ["A", "B"]],
                [2, MULTILINE_SUMMARY],
                [2, "A"],
                [2, "B"],
            ]
        ]
    )

    rows = await _run(engine, _req(["A"]), result)

    assert rows[0].fact == "First meaningful line"


@pytest.mark.parametrize("summary", BLANK_SUMMARIES)
async def test_blank_summary_rows_are_skipped(summary):
    engine = _engine()
    result = _compact_result(
        [
            [
                [2, "MENTIONS"],
                [3, 1],
                [6, ["A", "B"]],
                [2, summary],
                [2, "A"],
                [2, "B"],
            ]
        ]
    )

    assert await _run(engine, _req(["A"]), result) == []


async def test_non_str_summary_is_skipped():
    engine = _engine()
    result = _compact_result(
        [
            [
                [2, "MENTIONS"],
                [3, 1],
                [6, ["A", "B"]],
                [3, 42],  # summary is an int, not a str
                [2, "A"],
                [2, "B"],
            ]
        ]
    )

    assert await _run(engine, _req(["A"]), result) == []


async def test_missing_edge_type_defaults_to_relates_to():
    engine = _engine()
    result = _compact_result(
        [
            [
                [5, None],  # edge_type null
                [3, 1],
                [6, ["A", "B"]],
                [2, "Some fact"],
                [2, "A"],
                [2, "B"],
            ]
        ]
    )

    rows = await _run(engine, _req(["A"]), result)

    assert rows[0].edge_type == "RELATES_TO"


async def test_null_hop_defaults_to_one():
    engine = _engine()
    result = _compact_result(
        [
            [
                [2, "MENTIONS"],
                [5, None],  # hop null
                [6, ["A", "B"]],
                [2, "Some fact"],
                [2, "A"],
                [2, "B"],
            ]
        ]
    )

    rows = await _run(engine, _req(["A"]), result)

    assert rows[0].hop_distance == 1


async def test_source_and_target_fall_back_to_path_nodes():
    """With null seed/target columns, source = path[0] and target = ''."""
    engine = _engine()
    result = _compact_result(
        [
            [
                [2, "MENTIONS"],
                [3, 1],
                [6, ["A", "B", "C"]],
                [2, "Some fact"],
                [5, None],  # seed_in_path null
                [5, None],  # target_in_path null
            ]
        ]
    )

    rows = await _run(engine, _req(["A"]), result)

    assert rows[0].source_entity == "A"
    assert rows[0].target_entity == ""


async def test_empty_path_yields_empty_source():
    engine = _engine()
    result = _compact_result(
        [
            [
                [2, "MENTIONS"],
                [3, 1],
                [6, []],  # empty path
                [2, "Some fact"],
                [5, None],
                [5, None],
            ]
        ]
    )

    rows = await _run(engine, _req(["A"]), result)

    assert rows[0].source_entity == ""
    assert rows[0].path == []


async def test_short_row_is_skipped_not_fatal():
    """A row with fewer than 6 columns raises on unpack and is caught per-row."""
    engine = _engine()
    result = _compact_result(
        [
            [
                [2, "MENTIONS"],
                [3, 1],
                [6, ["A", "B"]],
                [2, "truncated"],
            ]
        ]
    )

    assert await _run(engine, _req(["A"]), result) == []


async def test_non_numeric_hop_is_skipped():
    engine = _engine()
    result = _compact_result(
        [
            [
                [2, "MENTIONS"],
                [2, "not-a-number"],  # int("not-a-number") raises
                [6, ["A", "B"]],
                [2, "Some fact"],
                [2, "A"],
                [2, "B"],
            ]
        ]
    )

    assert await _run(engine, _req(["A"]), result) == []


async def test_bad_row_does_not_discard_good_rows():
    """Per-row try/except: one malformed row is dropped, the rest survive."""
    engine = _engine()
    result = _compact_result(
        [
            [  # bad: short row
                [2, "MENTIONS"],
                [3, 1],
            ],
            [  # good
                [2, "MENTIONS"],
                [3, 1],
                [6, ["A", "B"]],
                [2, "Good fact"],
                [2, "A"],
                [2, "B"],
            ],
            [  # bad: blank summary
                [2, "MENTIONS"],
                [3, 1],
                [6, ["A", "B"]],
                [2, "   "],
                [2, "A"],
                [2, "B"],
            ],
            [  # good
                [2, "CODE"],
                [3, 2],
                [6, ["B", "C"]],
                [2, "Another good fact"],
                [2, "B"],
                [2, "C"],
            ],
        ]
    )

    rows = await _run(engine, _req(["A", "B"]), result)

    assert [r.fact for r in rows] == ["Good fact", "Another good fact"]
    assert [r.edge_type for r in rows] == ["MENTIONS", "CODE"]


# ─── degenerate results ─────────────────────────────────────────────────


async def test_empty_rows_yields_empty_list():
    engine = _engine()
    assert await _run(engine, _req(["A"]), _compact_result([])) == []


@pytest.mark.parametrize("result", [None, [], "not-a-list", {}])
async def test_non_result_shapes_yield_empty_list(result):
    """``_parse_falkor_rows`` filters these out before the row loop."""
    engine = _engine()
    assert await _run(engine, _req(["A"]), result) == []


async def test_rows_key_not_a_list_yields_empty_list():
    engine = _engine()
    engine._redis = MagicMock()
    engine._redis.execute_command = AsyncMock(return_value=[[[1, "c0"]], "not-a-list-of-rows", []])

    assert await engine._cypher_bfs(_req(["A"])) == []
