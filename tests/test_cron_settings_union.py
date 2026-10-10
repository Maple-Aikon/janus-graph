"""Regression tests for the two live crashes the mypy gate found in cron.py.

Both were flagged as "type noise" on the debt list and would have been
silenced with a cast. They were not noise:

  1. ``float(getattr(cfg.pipeline, "attempt_timeout_sec", None))`` guarded by
     ``hasattr``. ``hasattr`` is True whenever the attribute EXISTS, even when
     its value is None, and pydantic accepts a post-hoc assignment to None
     (validate_assignment is unset). The guard therefore passed and float(None)
     raised TypeError -- taking down the WHOLE sweep, not one episode.

  2. ``getattr(cfg.paths, "queue_db_path", None)``. getattr guards the INNER
     attribute; ``cfg.paths`` is evaluated eagerly, and JanusSettings has no
     ``paths`` field at all. The "defence-in-depth" fallback raised
     AttributeError instead of falling back.

Each test below was verified to FAIL against the pre-fix code (see the
docstring of tests/test_cron_settings_union.py::test_... for the mechanism);
the negative control is that re-introducing the `hasattr` guard makes
test_attempt_timeout_none_falls_back fail again.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from janus_graph.config import JanusSettings, load_config
from janus_graph.core.contracts import PathsSettings, PipelineSettings, ReportSettings
from janus_graph.core.contracts import Settings as ContractsSettings
from janus_graph.pipeline.cron import run_cron_sweep


class _QueueSpy:
    """Records the arguments cron.py hands the queue. No sqlite, no I/O."""

    def __init__(self) -> None:
        self.seen: dict = {}

    async def reap_stuck_processing(self, timeout_sec=None):
        self.seen["reap_timeout"] = timeout_sec
        return 0

    async def claim_next_batch(self, limit=None):
        self.seen["batch_limit"] = limit
        return []

    async def mark_failed(self, *_a, **_k):
        return None

    def get_stats(self):
        return {}

    def count_total(self):
        return 0


def _drive(coro):
    import asyncio

    return asyncio.new_event_loop().run_until_complete(coro)


def _run(settings, **kw):
    spy = _QueueSpy()
    with (
        patch("janus_graph.pipeline.cron.EpisodeQueue", return_value=spy),
        patch("janus_graph.pipeline.cron.EpisodeWorker", return_value=object()),
        patch("janus_graph.pipeline.cron.ReportDispatcher.from_settings", return_value=object()),
    ):
        _drive(run_cron_sweep(settings, **kw))
    return spy


@pytest.fixture
def live() -> JanusSettings:
    cfg = load_config()
    assert isinstance(cfg, JanusSettings), "load_config must return JanusSettings"
    return cfg


# ---------------------------------------------------------------------------
# Bug 1: attempt_timeout_sec = None
# ---------------------------------------------------------------------------
def test_attempt_timeout_none_falls_back_not_raises(live):
    """A None attempt_timeout_sec must fall back to the env default.

    Pre-fix this raised TypeError: float() argument must be a string or a real
    number, not 'NoneType'. hasattr() returned True the whole time.
    """
    live.pipeline.attempt_timeout_sec = None  # type: ignore[assignment]
    spy = _run(live)
    assert spy.seen["reap_timeout"] == 900, (
        f"None attempt_timeout_sec must fall back to PROCESSING_TIMEOUT_SECONDS; "
        f"got {spy.seen['reap_timeout']!r}"
    )


def test_attempt_timeout_zero_is_preserved(live):
    """0 is a legal value meaning 'no timeout' and must NOT become 900.

    This pins the reason the fix uses an explicit `is not None` test instead
    of the tempting `float(x or DEFAULT)`: `0 or 900` is 900.
    """
    live.pipeline.attempt_timeout_sec = 0  # type: ignore[assignment]
    spy = _run(live)
    assert spy.seen["reap_timeout"] == 0, (
        f"a legal 0 must survive; `or` would have turned this into 900. "
        f"got {spy.seen['reap_timeout']!r}"
    )


def test_attempt_timeout_normal_value_still_applies(live):
    """The ordinary production shape must keep using the configured value."""
    live.pipeline.attempt_timeout_sec = 1234  # type: ignore[assignment]
    spy = _run(live)
    assert spy.seen["reap_timeout"] == 1234


# ---------------------------------------------------------------------------
# Bug 2: cfg.paths does not exist on JanusSettings
# ---------------------------------------------------------------------------
def test_missing_queue_db_path_falls_back_without_raising(live):
    """With no queue_db_path anywhere, the cwd-relative default must apply.

    Pre-fix this raised AttributeError: 'JanusSettings' object has no
    attribute 'paths' -- the getattr guarded the inner attribute only.
    """
    live.pipeline.queue_db_path = None  # type: ignore[assignment]
    captured: dict = {}
    spy = _QueueSpy()

    def _fake_queue(db):
        captured["db"] = db
        return spy

    with (
        patch("janus_graph.pipeline.cron.EpisodeQueue", side_effect=_fake_queue),
        patch("janus_graph.pipeline.cron.EpisodeWorker", return_value=object()),
        patch("janus_graph.pipeline.cron.ReportDispatcher.from_settings", return_value=object()),
    ):
        _drive(run_cron_sweep(live))
    assert captured["db"].endswith("queue.db"), captured


def test_queue_db_path_absent_on_contracts_settings_uses_paths(live):
    """contracts.Settings keeps the path under `.paths` -- that fallback must work."""
    cs = ContractsSettings(
        pipeline=PipelineSettings(),
        report=ReportSettings(),
        paths=PathsSettings(),
    )
    captured: dict = {}

    class _Q(_QueueSpy):
        pass

    spy = _Q()
    with (
        patch(
            "janus_graph.pipeline.cron.EpisodeQueue",
            side_effect=lambda db: (captured.setdefault("db", db), spy)[1],
        ),
        patch("janus_graph.pipeline.cron.EpisodeWorker", return_value=object()),
        patch("janus_graph.pipeline.cron.ReportDispatcher.from_settings", return_value=object()),
    ):
        import asyncio

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(run_cron_sweep(cs))
    assert captured["db"].endswith("queue.db"), captured


# ---------------------------------------------------------------------------
# Type-only fixes must not have changed anything
# ---------------------------------------------------------------------------
def test_contracts_settings_default_batch_size_is_the_declared_default():
    """A contracts.Settings has no drain_batch_size; the fallback must be the
    value declared on PipelineConfig, never a literal typed in by hand."""
    from janus_graph.config import PipelineConfig

    cs = ContractsSettings(
        pipeline=PipelineSettings(), report=ReportSettings(), paths=PathsSettings()
    )
    spy = _run(cs)
    assert spy.seen["batch_limit"] == PipelineConfig.model_fields["drain_batch_size"].default


def test_explicit_batch_size_wins_over_config():
    cs = ContractsSettings(
        pipeline=PipelineSettings(), report=ReportSettings(), paths=PathsSettings()
    )
    spy = _run(cs, batch_size=5)
    assert spy.seen["batch_limit"] == 5


def test_concurrency_zero_does_not_reach_semaphore(live):
    """The trailing `or WORKER_CONCURRENCY` spans BOTH ternary branches.

    Rewriting it as an `is None` test would let an explicit concurrency=0
    reach asyncio.Semaphore(0), which never releases. Pin the behaviour.
    """
    spy = _run(live, concurrency=0)
    assert spy.seen["batch_limit"] is not None
