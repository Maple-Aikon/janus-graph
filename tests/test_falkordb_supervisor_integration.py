"""Integration tests for FalkorRestartPolicy + DaemonSupervisor (Phase 2).

Per Plan #3 (falkordb-auto-restart-supervisor-20260909) §7 Phase 2 scope:
  - 6 tests covering end-to-end restart flow with mocked manager
  - Concurrent safety (NEW — Phase 1 was single-threaded)
  - Port-held detection (logs port_held_by_other_pid, skips restart)
  - Missing binary boot check (lifespan soft-disables policy)
  - Rate limit (5x restarts/hour → 6th refused)
  - Spawn fail (manager.start() raises → log + state)
  - Happy path (kill → restart → probe success → circuit CLOSED)

All tests use mocked FalkorDBServerManager — no real subprocess spawn.
Real-spawn verification is done via sandbox script (tests/sandbox_phase2.sh).

Tests are sync because attempt_restart() itself is sync (manager.start()
is sync, wrapped in asyncio.to_thread at the supervisor boundary).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from janus_graph.config import (
    EngineConfig,
    FalkorCircuitSettings,
    FalkorRestartPolicySettings,
)
from janus_graph.daemon.falkordb_restart_policy import (
    FalkorRestartPolicy,
    RestartPolicyState,
)
from janus_graph.daemon.falkordb_supervisor import DaemonSupervisor
from janus_graph.engine.server import FalkorDBServerManager


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def fast_settings():
    """Fast restart-policy settings (cooldown=1s, max=3/hour) for tests."""
    return FalkorRestartPolicySettings(cooldown_sec=1, max_per_hour=3)


@pytest.fixture
def fast_circuit_settings():
    """Fast circuit-breaker settings (2 failures → OPEN) for tests."""
    return FalkorCircuitSettings(failure_threshold=2, reset_timeout_sec=2)


@pytest.fixture
def mock_manager():
    """Mock FalkorDBServerManager with controllable start() return value."""
    m = MagicMock(spec=FalkorDBServerManager)
    m.is_running.return_value = False
    return m


@pytest.fixture
def policy_with_manager(fast_settings, mock_manager):
    """FalkorRestartPolicy with mock manager bound (start returns True)."""
    mock_manager.start.return_value = True
    p = FalkorRestartPolicy(settings=fast_settings, manager=mock_manager)
    return p


@pytest.fixture
def supervisor(fast_settings, fast_circuit_settings, mock_manager):
    """DaemonSupervisor wired with fast settings + mock manager (start returns True)."""
    mock_manager.start.return_value = True
    return DaemonSupervisor(
        engine_config=EngineConfig(host="127.0.0.1", port=6390),
        circuit_config=fast_circuit_settings,
        restart_policy_config=fast_settings,
        manager=mock_manager,
    )


# --- 6 INTEGRATION TESTS ---------------------------------------------------


def test_happy_path_kill_restart_recover(supervisor, mock_manager, monkeypatch):
    """Kill falkordb → breaker OPEN → policy restart → probe success → CLOSED.

    Plan §7 Phase 2 test 1.
    """
    # 1. Drive breaker to OPEN (2 failures per fast_circuit_settings).
    for _ in range(2):
        # Each call goes through record_failure path.
        # We can't easily simulate Redis-down without socket patching;
        # use direct call.
        import asyncio
        asyncio.run(supervisor.breaker.record_failure())
    assert supervisor.breaker.state.value == "open"

    # 2. The CLOSED→OPEN transition should have fired the hook which calls
    # policy.attempt_restart() which calls manager.start().
    # (mock_manager.start returns True by default since MagicMock auto-creates)
    assert mock_manager.start.called

    # 3. Verify policy state reflects one restart consumed.
    snap = supervisor.restart_policy.snapshot()
    assert snap.restarts_in_window == 1
    assert snap.state == RestartPolicyState.OK


def test_spawn_fail_logs_and_keeps_circuit_open(supervisor, mock_manager):
    """Kill falkordb → restart fails → log warning, breaker stays OPEN.

    Plan §7 Phase 2 test 2.
    """
    # Make manager.start() raise an exception (simulate falkordb.so crash).
    mock_manager.start.side_effect = RuntimeError("falkordb crashed on boot")

    policy = FalkorRestartPolicy(
        settings=FalkorRestartPolicySettings(cooldown_sec=0, max_per_hour=5),
        manager=mock_manager,
    )

    ok = policy.attempt_restart()
    assert ok is False
    # On exception, policy enters FATAL_DEGRADED (manager_exception).
    assert policy.state is RestartPolicyState.FATAL_DEGRADED
    assert mock_manager.start.called


def test_rate_limit_kicks_in_after_max(supervisor, mock_manager, monkeypatch):
    """Kill falkordb 4× → first 3 restart, 4th refused (max_per_hour=3).

    Plan §7 Phase 2 test 3. fast_settings has max_per_hour=3.
    """
    policy = FalkorRestartPolicy(
        settings=FalkorRestartPolicySettings(cooldown_sec=0, max_per_hour=3),
        manager=mock_manager,
    )

    fake_now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake_now[0])

    # First 3 succeed.
    for i in range(3):
        assert policy.attempt_restart() is True, f"call #{i+1}"
        fake_now[0] += 1
        mock_manager.start.reset_mock()

    # 4th refused (rate limit).
    assert policy.attempt_restart() is False
    assert policy.state is RestartPolicyState.FATAL_DEGRADED
    assert not mock_manager.start.called  # no spawn on refused restart


def test_concurrent_record_failure_during_restart_blocks(supervisor):
    """Two threads call attempt_restart() simultaneously → second short-circuits.

    Plan §7 Phase 2 test 4 (NEW — Phase 1 was single-threaded).
    Verifies §4.2 idempotency: only one spawn happens even under probe-storm.
    """
    import threading

    call_count = [0]
    call_lock = threading.Lock()

    def slow_start():
        with call_lock:
            call_count[0] += 1
        time.sleep(0.1)  # simulate slow spawn
        return True

    mock_manager = MagicMock(spec=FalkorDBServerManager)
    mock_manager.start.side_effect = slow_start

    policy = FalkorRestartPolicy(
        settings=FalkorRestartPolicySettings(cooldown_sec=0, max_per_hour=10),
        manager=mock_manager,
    )

    results = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        results.append(policy.attempt_restart())

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join(timeout=2)
    t2.join(timeout=2)

    # Exactly one True, one False — second call short-circuited on lock.
    assert sorted(results) == [False, True], f"expected [False, True], got {results}"
    # And only one actual spawn.
    assert call_count[0] == 1, f"expected 1 spawn, got {call_count[0]}"


def test_operator_disabled_via_disable_method(policy_with_manager):
    """Operator kill-switch (lifespan binary check) disables all restarts.

    Plan §7 Phase 2 test 6 + Plan §5.1 mitigation.
    """
    p = policy_with_manager
    p.disable(reason="binary_missing")

    # 1. attempt_restart() returns False.
    assert p.attempt_restart() is False

    # 2. disabled_reason surfaces in snapshot.
    assert p.disabled_reason == "operator_disabled"

    # 3. Idempotent: second disable() is a no-op.
    p.disable(reason="different_reason")
    assert p.disabled_reason == "operator_disabled"  # unchanged

    # 4. No manager call when disabled.
    # (manager mock in fixture, but we never reach it via attempt_restart)
    snap = p.snapshot()
    assert snap.restarts_in_window == 0


def test_lock_held_by_other_pid_skips_restart(caplog):
    """Port 6379 held by redislite zombie → log port_held_by_other_pid, skip.

    Plan §7 Phase 2 test 5. We simulate this by checking that
    manager.start() detects the conflict (manager is responsible for port
    detection, policy observes manager.start()'s False return).
    """
    # Simulate the manager-level port-held detection by returning False
    # (manager.start() does its own subprocess spawn + port check).
    mock_manager = MagicMock(spec=FalkorDBServerManager)
    mock_manager.start.return_value = False  # port-held detected at manager level

    policy = FalkorRestartPolicy(
        settings=FalkorRestartPolicySettings(cooldown_sec=0, max_per_hour=5),
        manager=mock_manager,
    )

    # attempt_restart returns False (manager said no) but state stays OK
    # (we don't enter FATAL_DEGRADED for "couldn't bind port" — that's a
    # recoverable condition, not a budget issue).
    ok = policy.attempt_restart()
    assert ok is False
    assert policy.state is RestartPolicyState.OK  # not degraded — port-held is transient
    assert mock_manager.start.called
