"""MCP surface for DLQ observability: ``queue_status`` filters + ``graphiti_health`` breakdown.

Contract rule (plan wave 2): the DLQ numbers the operator sees must come
from ONE predicate. ``get_stats()['dlq']`` counts ``recovered_at IS NULL``;
the per-class breakdown must count the same set, or ``by_class`` and
``dlq`` describe different populations and neither number is trustworthy.

These tests use a fake queue (no live DB) on purpose: the wiring is what is
under test, not SQLite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from janus_graph.config import JanusSettings
from janus_graph.mcp.server import create_mcp_server


def _struct(res) -> dict:
    import json

    if getattr(res, "structured_content", None):
        return res.structured_content.get("result", res.structured_content)
    if getattr(res, "content", None) and len(res.content) > 0:
        return json.loads(res.content[0].text)
    return {}


async def _call(mcp, name: str, **kwargs) -> dict:
    return _struct(await mcp.call_tool(name, kwargs))


def _seed(queue, rows) -> None:
    with queue._get_connection() as conn:
        for i, (last_error, recovered) in enumerate(rows):
            conn.execute(
                "INSERT INTO dead_letter (episode_id, payload_json, last_error, "
                "attempt_count, failed_at, recovered_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    f"ep-{i:03d}",
                    '{"content": "x"}',
                    last_error,
                    3,
                    f"2026-10-{i + 1:02d}T00:00:00+00:00",
                    "2026-10-16T00:00:00+00:00" if recovered else None,
                ),
            )
        conn.commit()


@pytest.fixture
def cfg_with_taxonomy(temp_dir: Path) -> JanusSettings:
    """Config whose replay taxonomy is the exception-type pair used live."""
    c = JanusSettings()
    c.pipeline.queue_db_path = str(temp_dir / "episodes.db")
    c.pipeline.dlq_replay.classes = ["ValidationError", "TimeoutError"]
    return c


# ── queue_status: open_only is the fix for the two-populations bug ──────────


@pytest.mark.asyncio
async def test_dlq_recent_shows_only_open_rows(cfg_with_taxonomy: JanusSettings):
    """AC1: ``dlq_recent`` must not mix recovered rows into "recent errors".

    Before this change, ``stats.dlq`` (open) and ``dlq_recent`` (everything)
    described two different populations in the SAME response.
    """
    from janus_graph.pipeline.queue import EpisodeQueue

    mcp = create_mcp_server(cfg_with_taxonomy)
    q = EpisodeQueue(cfg_with_taxonomy.pipeline.queue_db_path)
    _seed(
        q,
        [
            ("ValidationError: open one", False),
            ("TimeoutError: open two", False),
            ("ValidationError: already recovered", True),
        ],
    )

    out = await _call(mcp, "queue_status")

    assert out["stats"]["dlq"] == 2
    assert {r["id"] for r in out["dlq_recent"]} == {"ep-000", "ep-001"}


@pytest.mark.asyncio
async def test_queue_status_accepts_classes_and_error_chars(cfg_with_taxonomy: JanusSettings):
    from janus_graph.pipeline.queue import EpisodeQueue

    mcp = create_mcp_server(cfg_with_taxonomy)
    q = EpisodeQueue(cfg_with_taxonomy.pipeline.queue_db_path)
    _seed(
        q,
        [
            ("ValidationError: " + "V" * 600, False),
            ("TimeoutError: <no message>", False),
            ("APIStatusError: Error code: 402", False),
        ],
    )

    out = await _call(
        mcp,
        "queue_status",
        classes=["ValidationError"],
        error_chars=200,
    )

    assert [r["id"] for r in out["dlq_recent"]] == ["ep-000"]
    assert len(out["dlq_recent"][0]["last_error"]) == 201


@pytest.mark.asyncio
async def test_queue_status_unknown_class_is_empty_not_everything(cfg_with_taxonomy: JanusSettings):
    from janus_graph.pipeline.queue import EpisodeQueue

    mcp = create_mcp_server(cfg_with_taxonomy)
    q = EpisodeQueue(cfg_with_taxonomy.pipeline.queue_db_path)
    _seed(q, [("ValidationError: only row", False)])

    out = await _call(mcp, "queue_status", classes=["không-có"])

    assert out["dlq_recent"] == []


@pytest.mark.asyncio
async def test_queue_status_keeps_the_old_keys(cfg_with_taxonomy: JanusSettings):
    """Adding arguments must not remove anything a caller already reads."""
    mcp = create_mcp_server(cfg_with_taxonomy)

    out = await _call(mcp, "queue_status")

    assert {"stats", "max_attempts", "dlq_recent"} <= set(out)


# ── graphiti_health: one predicate, so the numbers reconcile ────────────────


@pytest.mark.asyncio
async def test_health_dlq_block_reconciles_with_stats(cfg_with_taxonomy: JanusSettings):
    from janus_graph.pipeline.queue import EpisodeQueue

    mcp = create_mcp_server(cfg_with_taxonomy)
    q = EpisodeQueue(cfg_with_taxonomy.pipeline.queue_db_path)
    _seed(
        q,
        [
            ("ValidationError: one", False),
            ("ValidationError: two", False),
            ("TimeoutError: three", False),
            ("APIStatusError: Error code: 402", False),  # open, matches no class
            ("ValidationError: recovered", True),  # recovered, matches a class
        ],
    )

    out = await _call(mcp, "graphiti_health")

    dlq = out["queue_stats"]["dlq"]
    assert out["dlq"] == {
        "open": dlq,
        "by_class": {"ValidationError": 2, "TimeoutError": 1},
        "unclassified": 1,
    }


@pytest.mark.asyncio
async def test_health_by_class_is_the_replay_taxonomy(cfg_with_taxonomy: JanusSettings):
    """The taxonomy comes from ``DlqReplayConfig.classes`` — no second list."""
    from janus_graph.pipeline.queue import EpisodeQueue

    mcp = create_mcp_server(cfg_with_taxonomy)
    q = EpisodeQueue(cfg_with_taxonomy.pipeline.queue_db_path)
    _seed(q, [("NotFoundError: 404", False)])

    out = await _call(mcp, "graphiti_health")

    assert set(out["dlq"]["by_class"]) == {"ValidationError", "TimeoutError"}
    assert out["dlq"]["by_class"] == {"ValidationError": 0, "TimeoutError": 0}
    assert out["dlq"]["unclassified"] == 1, "the 404 row is open but unclassified"


@pytest.mark.asyncio
async def test_health_unclassified_is_zero_when_taxonomy_covers_everything(
    cfg_with_taxonomy: JanusSettings,
):
    from janus_graph.pipeline.queue import EpisodeQueue

    mcp = create_mcp_server(cfg_with_taxonomy)
    q = EpisodeQueue(cfg_with_taxonomy.pipeline.queue_db_path)
    _seed(q, [("ValidationError: only", False)])

    out = await _call(mcp, "graphiti_health")

    assert out["dlq"]["unclassified"] == 0


@pytest.mark.asyncio
async def test_health_without_taxonomy_reports_no_breakdown(cfg_with_taxonomy: JanusSettings):
    """Empty taxonomy ⇒ no breakdown, NOT "everything is unclassified"."""
    from janus_graph.pipeline.queue import EpisodeQueue

    cfg_with_taxonomy.pipeline.dlq_replay.classes = []
    mcp = create_mcp_server(cfg_with_taxonomy)
    q = EpisodeQueue(cfg_with_taxonomy.pipeline.queue_db_path)
    _seed(q, [("ValidationError: only", False)])

    out = await _call(mcp, "graphiti_health")

    assert out["dlq"] == {"open": 1}


@pytest.mark.asyncio
async def test_health_open_zero_with_rows_present_is_distinguishable(
    cfg_with_taxonomy: JanusSettings,
):
    """N1: ``open: 0`` with an all-recovered table must not be ambiguous.

    ``by_class`` all-zero and an empty table both print 0, so ``open`` is
    the field that tells them apart — that is the whole reason it is here.
    """
    from janus_graph.pipeline.queue import EpisodeQueue

    mcp = create_mcp_server(cfg_with_taxonomy)
    q = EpisodeQueue(cfg_with_taxonomy.pipeline.queue_db_path)
    _seed(q, [("ValidationError: recovered", True)])

    out = await _call(mcp, "graphiti_health")

    assert out["dlq"]["open"] == 0
    assert out["dlq"]["by_class"] == {"ValidationError": 0, "TimeoutError": 0}


@pytest.mark.asyncio
async def test_health_keeps_the_old_keys(cfg_with_taxonomy: JanusSettings):
    mcp = create_mcp_server(cfg_with_taxonomy)

    out = await _call(mcp, "graphiti_health")

    assert {"status", "version", "engine_reachable", "queue_stats"} <= set(out)
