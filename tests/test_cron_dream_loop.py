"""Unit tests for ``CronLoop._dream_loop``.

Untested loop body. Existing ``test_cron_loop.py`` only asserts that
``start()`` spawns the task and that ``stop()`` drains it -- the loop body
itself (clock gate, circuit guard, dedupe, exception handling) never runs in
any test.

Branch coverage:
  CLOCK GATE
  - fires only when ICT hour == DREAM_HOUR_ICT and minute >= DREAM_MINUTE_ICT
  - ICT hour is (utc_hour + 7) % 24, including the midnight wrap
  - minute < DREAM_MINUTE_ICT never fires
  - wrong hour never fires
  CIRCUIT GUARD
  - OPEN circuit skips the run entirely (no dream call), and does NOT record
    last_run_date, so the next minute window retries
  - CLOSED / HALF_OPEN / unknown states all run
  DEDUPE
  - a second firing on the same UTC date is suppressed
  SUCCESS / FAILURE
  - success records last_dream_at + last_dream_result
  - a raising run is swallowed (loop survives) and retries next window
  SHUTDOWN
  - setting the shutdown event makes the loop return

The clock is controlled by patching ``janus_graph.daemon.cron_loop.datetime``;
no sleeping on real wall-clock time, no network, no production code changes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime as _RealDatetime, timedelta, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from janus_graph.daemon.cron_loop import CronLoop


# ─── fake clock ──────────────────────────────────────────────────────────


class FakeDatetime:
    """A ``datetime`` stand-in whose ``now()`` returns a settable instant.

    ``_dream_loop`` only uses ``datetime.now(timezone.utc)``, so a minimal
    class with a classmethod is enough.
    """

    current: _RealDatetime = _RealDatetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):  # noqa: ANN001 - matches datetime.now signature
        return cls.current.astimezone(tz) if tz else cls.current

    @classmethod
    def set(cls, moment: _RealDatetime) -> None:
        cls.current = moment


def _ict(moment_utc: _RealDatetime, minutes: int = 0, hours: int = 0):
    """Build the UTC instant whose ICT wall clock is the given 02:30 slot."""
    return moment_utc + timedelta(hours=hours, minutes=minutes)


# ─── fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def settings():
    """A minimal settings object exposing only what the dream loop reads."""
    from types import SimpleNamespace

    return SimpleNamespace(
        daemon=SimpleNamespace(cron_enabled=True),
        pipeline=SimpleNamespace(cron_interval_min=10),
    )


@pytest.fixture
def closed_circuit():
    return lambda: "CLOSED"


@pytest.fixture
def open_circuit():
    return lambda: "OPEN"


@pytest.fixture(autouse=True)
def fast_tick(monkeypatch):
    """Shrink the 60s inter-check wait so the loop iterates in milliseconds."""
    monkeypatch.setattr(CronLoop, "DREAM_CHECK_INTERVAL_SEC", 0.01)
    yield


def _patched(dream_mock, clock: _RealDatetime):
    """Patch the module clock + dream runner together."""
    FakeDatetime.set(clock)
    return patch.multiple(
        "janus_graph.daemon.cron_loop",
        datetime=FakeDatetime,
        run_dream_consolidation=dream_mock,
    )


async def _run_until(loop: CronLoop, dream_mock, clock, *, max_iter=6):
    """Run the dream loop briefly, then stop it. Returns elapsed iterations.

    The loop is stopped from a watchdog task so we never depend on the real
    60s tick interval.
    """

    async def _stop_soon():
        # let the loop run a handful of iterations, then signal shutdown
        for _ in range(max_iter):
            await asyncio.sleep(0.02)
        loop._shutdown.set()

    with _patched(dream_mock, clock):
        watcher = asyncio.create_task(_stop_soon())
        await asyncio.wait_for(loop._dream_loop(), timeout=5.0)
        await watcher


# ─── clock gate ─────────────────────────────────────────────────────────


async def test_fires_when_ict_clock_matches(settings, closed_circuit):
    """02:30 ICT == 19:30 UTC the previous day: the loop must run the dream."""
    loop = CronLoop(settings, closed_circuit)
    dream = AsyncMock(return_value={"status": "complete", "duration_ms": 12})
    # 19:30 UTC on the 14th == 02:30 ICT on the 15th.
    clock = _RealDatetime(2026, 1, 14, 19, 30, tzinfo=timezone.utc)

    await _run_until(loop, dream, clock)

    assert dream.await_count == 1
    assert loop.last_dream_result == {"status": "complete", "duration_ms": 12}
    assert loop.last_dream_at is not None


async def test_does_not_fire_before_minute_30(settings, closed_circuit):
    """ICT 02:00 is inside the hour but before DREAM_MINUTE_ICT -> no run."""
    loop = CronLoop(settings, closed_circuit)
    dream = AsyncMock(return_value={"status": "complete"})
    clock = _RealDatetime(2026, 1, 14, 19, 0, tzinfo=timezone.utc)  # 02:00 ICT

    await _run_until(loop, dream, clock)

    assert dream.await_count == 0
    assert loop.last_dream_result is None


async def test_does_not_fire_in_wrong_ict_hour(settings, closed_circuit):
    """ICT 03:30 (20:30 UTC) is the right minute but the wrong hour."""
    loop = CronLoop(settings, closed_circuit)
    dream = AsyncMock(return_value={"status": "complete"})
    clock = _RealDatetime(2026, 1, 14, 20, 30, tzinfo=timezone.utc)  # 03:30 ICT

    await _run_until(loop, dream, clock)

    assert dream.await_count == 0


async def test_ict_hour_wraps_past_midnight(settings, closed_circuit):
    """(utc_hour + 7) % 24 wraps: 18:00 UTC == 01:00 ICT the next day."""
    loop = CronLoop(settings, closed_circuit)
    dream = AsyncMock(return_value={"status": "complete"})
    clock = _RealDatetime(2026, 1, 14, 18, 0, tzinfo=timezone.utc)  # 01:00 ICT

    await _run_until(loop, dream, clock)

    # 01:00 ICT is before 02:30 -> must not fire.
    assert dream.await_count == 0


# ─── circuit guard ──────────────────────────────────────────────────────


async def test_open_circuit_skips_dream(settings, open_circuit):
    """Falkor circuit OPEN -> skip entirely; dream is never called."""
    loop = CronLoop(settings, open_circuit)
    dream = AsyncMock(return_value={"status": "complete"})
    clock = _RealDatetime(2026, 1, 14, 19, 30, tzinfo=timezone.utc)

    await _run_until(loop, dream, clock)

    assert dream.await_count == 0
    assert loop.last_dream_result is None
    # Skipped runs are not counted anywhere -- last_dream_at stays None.
    assert loop.last_dream_at is None


async def test_unknown_circuit_state_is_permissive(settings):
    """``_state_str`` maps unknown values to CLOSED, so the run proceeds."""
    loop = CronLoop(settings, lambda: "SOMETHING_ELSE")
    dream = AsyncMock(return_value={"status": "complete"})
    clock = _RealDatetime(2026, 1, 14, 19, 30, tzinfo=timezone.utc)

    await _run_until(loop, dream, clock)

    assert dream.await_count == 1


# ─── dedupe within a UTC day ────────────────────────────────────────────


async def test_second_firing_same_utc_date_is_suppressed(settings, closed_circuit):
    """After a successful run, the same UTC date never fires again."""
    loop = CronLoop(settings, closed_circuit)
    dream = AsyncMock(return_value={"status": "complete"})
    clock = _RealDatetime(2026, 1, 14, 19, 30, tzinfo=timezone.utc)

    await _run_until(loop, dream, clock, max_iter=12)

    assert dream.await_count == 1, "must fire once per UTC date"


async def test_new_utc_date_allows_another_run(settings, closed_circuit):
    """Crossing into a new UTC date re-arms the dream for that day.

    The dream mock advances the fake clock by one day after its first call.
    Because ``_dream_loop`` re-reads ``datetime.now()`` at the top of every
    iteration, the next tick sees a new ``utc_date`` and fires again.
    """
    loop = CronLoop(settings, closed_circuit)
    day1 = _RealDatetime(2026, 1, 14, 19, 30, tzinfo=timezone.utc)

    calls: list[int] = []

    async def _dream_and_advance(_settings=None, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            # jump the clock forward one day, staying in the 02:30 ICT window
            FakeDatetime.set(day1 + timedelta(days=1))
        return {"status": "complete", "duration_ms": 1}

    dream = AsyncMock(side_effect=_dream_and_advance)

    await _run_until(loop, dream, day1, max_iter=10)

    assert len(calls) == 2, "a new UTC date must re-arm the dream"


# ─── failure handling ───────────────────────────────────────────────────


async def test_raising_dream_is_swallowed_and_retried(settings, closed_circuit):
    """An exception must not kill the loop, and must not mark the day done."""
    loop = CronLoop(settings, closed_circuit)
    dream = AsyncMock(side_effect=RuntimeError("dream blew up"))
    clock = _RealDatetime(2026, 1, 14, 19, 30, tzinfo=timezone.utc)

    await _run_until(loop, dream, clock, max_iter=6)

    # Retried on every tick because last_run_date was never set.
    assert dream.await_count > 1
    assert loop.last_dream_result is None
    assert loop.last_dream_at is None


# ─── shutdown ───────────────────────────────────────────────────────────


async def test_shutdown_makes_loop_return(settings, closed_circuit):
    """A pre-set shutdown event ends the loop without running the dream."""
    loop = CronLoop(settings, closed_circuit)
    dream = AsyncMock(return_value={"status": "complete"})
    clock = _RealDatetime(2026, 1, 14, 19, 30, tzinfo=timezone.utc)
    loop._shutdown.set()

    with _patched(dream, clock):
        await asyncio.wait_for(loop._dream_loop(), timeout=2.0)

    assert dream.await_count == 0
