"""Tests for FalkorRestartPolicy + circuit-breaker CLOSED->OPEN hook (Phase 1).

Plan #3 Section 7 Phase 1 verification: 11 tests covering:
  - 4 rate-limiter tests (cooldown, max_per_hour, snapshot, FATAL_DEGRADED)
  - 3 transition-hook tests (CLOSED->OPEN fires; HALF_OPEN->OPEN does NOT)
  - 3 -LOADING parsing tests (probe returns WARMING_UP, does not trip breaker)
  - 1 subprocess timeout test (TimeoutExpired -> False + circuit failure)

Per Plan #3 Section 9 Q7: the hook fires ONLY on CLOSED->OPEN (not HALF_OPEN->OPEN).
Per Plan #3 Section 9 HIGH #1: WARMING_UP (Redis -LOADING) does NOT trip breaker.
Per Plan #3 Section 9 BLOCK #2a: subprocess.run with timeout=10 returns False on
TimeoutExpired.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from janus_graph.config import (
    FalkorCircuitSettings,
    FalkorRestartPolicySettings,
)
from janus_graph.daemon.circuit_breaker import (
    CircuitState,
    FalkorCircuitBreaker,
)
from janus_graph.daemon.falkordb_restart_policy import (
    FalkorRestartPolicy,
    RestartPolicyState,
)
from janus_graph.daemon.falkordb_supervisor import (
    DaemonSupervisor,
    ProbeState,
    _redis_ping,
)
from janus_graph.engine.server import FalkorDBServerManager


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def fast_policy():
    settings = FalkorRestartPolicySettings(cooldown_sec=2, max_per_hour=3)
    return FalkorRestartPolicy(settings=settings)


@pytest.fixture
def fast_breaker():
    settings = FalkorCircuitSettings(failure_threshold=3, reset_timeout_sec=2)
    return FalkorCircuitBreaker(settings)


@pytest.fixture
def mock_clock(monkeypatch):
    fake_now = [0.0]

    def _now() -> float:
        return fake_now[0]

    def _advance(seconds: float) -> None:
        fake_now[0] += seconds

    monkeypatch.setattr(time, "monotonic", _now)
    return _advance


@pytest.fixture
def mock_manager():
    m = MagicMock(spec=FalkorDBServerManager)
    m.start.return_value = True
    m.is_running.return_value = True
    return m


# --- 4 RATE-LIMITER TESTS ---------------------------------------------------


def test_cooldown_blocks_second_restart(fast_policy, mock_manager):
    fast_policy.register_manager(mock_manager)
    assert fast_policy.attempt_restart() is True
    assert fast_policy.state is RestartPolicyState.OK
    assert fast_policy.attempt_restart() is False
    assert fast_policy.state is RestartPolicyState.COOLING_DOWN


def test_max_per_hour_enforced(fast_policy, mock_manager, monkeypatch):
    fast_policy.register_manager(mock_manager)
    fake_now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake_now[0])

    for _ in range(3):
        assert fast_policy.attempt_restart() is True
        fake_now[0] += 10

    assert fast_policy.attempt_restart() is False
    assert fast_policy.state is RestartPolicyState.FATAL_DEGRADED


def test_snapshot_reports_restarts_in_window(fast_policy, mock_manager):
    fast_policy.register_manager(mock_manager)
    snap_before = fast_policy.snapshot()
    assert snap_before.restarts_in_window == 0
    assert snap_before.state is RestartPolicyState.OK

    fast_policy.attempt_restart()
    snap_after = fast_policy.snapshot()
    assert snap_after.restarts_in_window == 1
    assert snap_after.last_restart_at is not None
    assert snap_after.cooldown_sec == 2


def test_fatal_degraded_callback_fires_once(mock_manager, monkeypatch):
    fired = []

    def cb(snap):
        fired.append(snap)

    policy = FalkorRestartPolicy(
        settings=FalkorRestartPolicySettings(cooldown_sec=0, max_per_hour=2),
        manager=mock_manager,
        on_budget_exhausted=cb,
    )

    fake_now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake_now[0])

    assert policy.attempt_restart() is True
    fake_now[0] += 1
    assert policy.attempt_restart() is True
    fake_now[0] += 1
    assert policy.attempt_restart() is False
    assert policy.state is RestartPolicyState.FATAL_DEGRADED
    assert len(fired) == 1

    fake_now[0] += 1
    assert policy.attempt_restart() is False
    assert len(fired) == 1


# --- 3 TRANSITION-HOOK TESTS ------------------------------------------------


async def test_closed_to_open_fires_callback(fast_breaker):
    fired = []

    def cb(snap, _mgr_sentinel):
        fired.append(snap.state)

    fast_breaker.on_open(cb)

    for _ in range(3):
        await fast_breaker.record_failure()
    assert fast_breaker.state is CircuitState.OPEN
    assert len(fired) == 1
    assert fired[0] is CircuitState.OPEN


async def test_half_open_to_open_does_not_fire_callback(fast_breaker, mock_clock):
    fired = []

    def cb(snap, _mgr_sentinel):
        fired.append(snap.state)

    fast_breaker.on_open(cb)

    for _ in range(3):
        await fast_breaker.record_failure()
    mock_clock(fast_breaker.settings.reset_timeout_sec + 0.1)
    assert await fast_breaker.allow_probe() is True

    # Reset fired list — we're now testing the HALF_OPEN->OPEN transition only.
    fired.clear()

    await fast_breaker.record_failure()
    assert fast_breaker.state is CircuitState.OPEN
    assert len(fired) == 0  # no restart storm from HALF_OPEN->OPEN


async def test_callback_exception_does_not_break_state_machine(fast_breaker):
    def bad_cb(snap, _mgr):
        raise RuntimeError("boom")

    fast_breaker.on_open(bad_cb)
    for _ in range(3):
        await fast_breaker.record_failure()

    assert fast_breaker.state is CircuitState.OPEN
    snap = fast_breaker.snapshot()
    assert snap.state is CircuitState.OPEN
    assert snap.failure_count == 3


# --- 3 -LOADING PARSING TESTS -----------------------------------------------


def test_redis_ping_loading_response_returns_warming_up():
    fake_sock = MagicMock()
    fake_sock.recv.return_value = b"-LOADING Redis is loading the dataset in memory\\"
    fake_sock.__enter__ = MagicMock(return_value=fake_sock)
    fake_sock.__exit__ = MagicMock(return_value=False)

    with patch("socket.create_connection", return_value=fake_sock):
        result = _redis_ping("127.0.0.1", 6379, 0.1)
    assert result is ProbeState.WARMING_UP


def test_redis_ping_dead_on_connection_refused():
    with patch(
        "socket.create_connection",
        side_effect=ConnectionRefusedError("nope"),
    ):
        result = _redis_ping("127.0.0.1", 1, 0.1)
    assert result is ProbeState.DEAD


async def test_probe_warming_up_does_not_trip_breaker(fast_breaker):
    manager = MagicMock(spec=FalkorDBServerManager)
    manager.is_running.return_value = True
    supervisor = DaemonSupervisor(
        engine_config=MagicMock(host="127.0.0.1", port=1),
        circuit_config=FalkorCircuitSettings(failure_threshold=2),
        manager=manager,
    )
    supervisor._breaker = fast_breaker
    assert fast_breaker.failure_count == 0

    async def fake_warming():
        return ProbeState.WARMING_UP
    supervisor._probe_redis = fake_warming  # type: ignore[assignment]

    await supervisor.probe()
    assert fast_breaker.failure_count == 0
    assert fast_breaker.state is CircuitState.CLOSED


# --- 1 SUBPROCESS TIMEOUT TEST ---------------------------------------------


def test_subprocess_timeout_returns_false(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    log_file = tmp_path / "log"
    pid_file = tmp_path / "pid"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    # Create fake binary files so resolve_binary_paths() does not raise.
    (bin_dir / "redis-server").write_bytes(b"")
    (bin_dir / "falkordb.so").write_bytes(b"")

    manager = FalkorDBServerManager(
        config=MagicMock(
            bin_dir=str(bin_dir),
            data_dir=str(data_dir),
            log_file=str(log_file),
            pid_file=str(pid_file),
            port=6379,
            host="127.0.0.1",
        )
    )

    # Mock is_running to return False so start() proceeds past the early-return.
    monkeypatch.setattr(
        FalkorDBServerManager,
        "is_running",
        lambda self: False,
    )

    # Force subprocess.run to raise TimeoutExpired (binary stuck).
    import janus_graph.engine.server as server_mod

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="redis-server", timeout=10)

    monkeypatch.setattr(server_mod.subprocess, "run", fake_run)
    assert manager.start() is False
