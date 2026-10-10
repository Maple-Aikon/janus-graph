"""Regression tests for the worker's reference_time fallback (P2, 2026-10-10).

``worker._execute_ingestion`` parsed ``record.created_at`` inside a bare
``except Exception: pass``. On an unparseable value it silently substituted
``datetime.now()`` -- ingesting the episode while stamping it with ingest time
instead of creation time, corrupting temporal attribution with no signal.

The fix adds a warning and nothing else. These tests pin BOTH halves of that
claim, because "it now logs" and "it still ingests the same way" can diverge:

  * the warning fires with the episode id (actionable -- which row is wrong?)
  * the record is still marked done and add_episode is still awaited once
    (observability-only change; a narrowing of the except clause would have
    turned a non-str created_at into a hard failure -- this test pins that it
    still does NOT)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from janus_graph.pipeline.queue import EpisodeRecord
from janus_graph.pipeline.worker import EpisodeWorker

BAD_TIMESTAMP = "not-a-real-timestamp"


class _StubQueue:
    """Minimal queue double: records calls, does no I/O."""

    def __init__(self) -> None:
        self.done: list[str] = []
        self.checkpoints: list[tuple[str, str]] = []

    async def update_checkpoint(self, ep_id: str, checkpoint: str) -> None:
        self.checkpoints.append((ep_id, checkpoint))

    async def mark_done(self, ep_id: str) -> None:
        self.done.append(ep_id)

    async def mark_failed(self, ep_id: str, err: str) -> None:  # pragma: no cover
        raise AssertionError("record must not fail on a bad timestamp")


def _record(created_at: object) -> EpisodeRecord:
    return EpisodeRecord(
        id="ep_test_0001",
        payload={"name": "n", "content": "a durable fact"},
        created_at=created_at,  # type: ignore[arg-type]
    )


class _StubGraphiti:
    group_id = "graphiti_memory"


class _StubSettings:
    """Stand-in for JanusSettings. `_execute_ingestion` reads
    `settings.graphiti.group_id` unconditionally (line 55), so the stub must
    carry a real graphiti section -- `object()` and `graphiti = None` both blow
    up before add_episode is ever reached."""

    graphiti = _StubGraphiti()


def _worker_with_mock_client(queue: _StubQueue) -> tuple[EpisodeWorker, AsyncMock]:
    worker = EpisodeWorker(queue, settings=_StubSettings())  # type: ignore[arg-type]
    client = AsyncMock()
    client.add_episode = AsyncMock(return_value=None)
    return worker, client


@pytest.mark.asyncio
async def test_unparseable_created_at_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    queue = _StubQueue()
    worker, client = _worker_with_mock_client(queue)

    with (
        caplog.at_level(logging.WARNING, logger="janus_graph.pipeline.worker"),
        patch("janus_graph.core.instance.create_graphiti_instance", return_value=client),
    ):
        ok = await worker.process_record(_record(BAD_TIMESTAMP))

    assert ok is True
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "bad created_at must not be swallowed silently"
    # The warning must identify WHICH episode is wrong, otherwise it is not
    # actionable -- a bare "invalid timestamp" is unactionable in a queue.
    assert any("ep_test_0001" in r.getMessage() for r in warnings)
    assert any(BAD_TIMESTAMP in r.getMessage() for r in warnings)


@pytest.mark.asyncio
async def test_unparseable_created_at_still_ingests() -> None:
    """Observability only: the episode must still be ingested and marked done."""
    queue = _StubQueue()
    worker, client = _worker_with_mock_client(queue)

    with patch("janus_graph.core.instance.create_graphiti_instance", return_value=client):
        ok = await worker.process_record(_record(BAD_TIMESTAMP))

    assert ok is True
    client.add_episode.assert_awaited_once()
    assert queue.done == ["ep_test_0001"]


@pytest.mark.asyncio
async def test_valid_created_at_is_used_verbatim_and_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fallback must not fire when the timestamp parses -- no false alarms."""
    queue = _StubQueue()
    worker, client = _worker_with_mock_client(queue)
    real = "2026-09-01T08:30:00+00:00"

    with (
        caplog.at_level(logging.WARNING, logger="janus_graph.pipeline.worker"),
        patch("janus_graph.core.instance.create_graphiti_instance", return_value=client),
    ):
        ok = await worker.process_record(_record(real))

    assert ok is True
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    kwargs = client.add_episode.await_args.kwargs
    assert kwargs["reference_time"] == datetime.fromisoformat(real)


@pytest.mark.asyncio
async def test_missing_created_at_uses_now_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """created_at=None is normal (nullable column), NOT a parse failure."""
    queue = _StubQueue()
    worker, client = _worker_with_mock_client(queue)

    with (
        caplog.at_level(logging.WARNING, logger="janus_graph.pipeline.worker"),
        patch("janus_graph.core.instance.create_graphiti_instance", return_value=client),
    ):
        ok = await worker.process_record(_record(None))

    assert ok is True
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    kwargs = client.add_episode.await_args.kwargs
    assert kwargs["reference_time"] > datetime(2020, 1, 1, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_non_string_created_at_still_falls_back() -> None:
    """Guards against over-narrowing the except clause to ValueError.

    An int created_at raises TypeError. Narrowing the handler would have turned
    today's silent fallback into a hard failure for migrated rows -- a
    behaviour change dressed up as a logging improvement.
    """
    queue = _StubQueue()
    worker, client = _worker_with_mock_client(queue)

    with patch("janus_graph.core.instance.create_graphiti_instance", return_value=client):
        ok = await worker.process_record(_record(1756721400))

    assert ok is True
    assert queue.done == ["ep_test_0001"]
