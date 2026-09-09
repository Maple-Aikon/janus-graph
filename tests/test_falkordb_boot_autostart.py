"""Tests for Phase 3.1 fix: boot-time auto-start when redislite zombie fooled PING.

Per Plan #3 Phase 3.1 - discovered during v0.7.0-restart-supervisor production
deploy (2026-09-10). The original probe() only checks Redis PING; a bare
redis-server (redislite orphan) responds to PING with +PONG but has no falkordb
module loaded. The new probe_strict() chains PING + MODULE LIST so a bare
redis gets correctly classified as DEAD and the boot path triggers start().
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from janus_graph.config import (
    FalkorCircuitSettings,
    FalkorRestartPolicySettings,
)
from janus_graph.daemon.falkordb_supervisor import (
    DaemonSupervisor,
    ProbeState,
    _module_list,
)


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def supervisor():
    """DaemonSupervisor with a mocked manager (no real subprocess spawn)."""
    engine_config = MagicMock()
    engine_config.host = "127.0.0.1"
    engine_config.port = 1
    circuit_config = FalkorCircuitSettings(failure_threshold=3)
    restart_policy_config = FalkorRestartPolicySettings(cooldown_sec=0, max_per_hour=5)
    manager = MagicMock()
    return DaemonSupervisor(
        engine_config=engine_config,
        circuit_config=circuit_config,
        restart_policy_config=restart_policy_config,
        manager=manager,
    )


def _fake_sock(recv_data):
    """Build a context-manager-friendly MagicMock socket with given recv data."""
    fake = MagicMock()
    fake.recv.return_value = recv_data
    fake.__enter__ = MagicMock(return_value=fake)
    fake.__exit__ = MagicMock(return_value=False)
    return fake


def _resp_module_list(module_names):
    """Build a fake RESP MODULE LIST reply (bytes) for the given module names.

    Uses bytes assembly with bytes([10]) so we never write raw newlines in source strings.
    """
    parts = []
    parts.append(b"*")
    parts.append(str(len(module_names)).encode("ascii"))
    parts.append(bytes([10]))
    for name in module_names:
        parts.append(b"*2")
        parts.append(bytes([10]))
        parts.append(b"$4")
        parts.append(bytes([10]))
        parts.append(b"name")
        parts.append(bytes([10]))
        parts.append(b"$")
        parts.append(str(len(name)).encode("ascii"))
        parts.append(bytes([10]))
        parts.append(name.encode("ascii"))
        parts.append(bytes([10]))
    return b"".join(parts)


# --- _module_list unit tests -----------------------------------------------


def test_module_list_returns_true_when_falkordb_loaded():
    """MODULE LIST reply containing "falkordb" -> True."""
    payload = _resp_module_list(["falkordb", "search", "times", "graph"])
    fake = _fake_sock(payload)
    with patch("socket.create_connection", return_value=fake):
        result = _module_list("127.0.0.1", 6379, 0.5)
    assert result is True


def test_module_list_returns_false_when_bare_redis_redislite_zombie():
    """MODULE LIST reply missing falkordb (redislite zombie) -> False."""
    payload = _resp_module_list([])
    fake = _fake_sock(payload)
    with patch("socket.create_connection", return_value=fake):
        result = _module_list("127.0.0.1", 6379, 0.5)
    assert result is False


def test_module_list_returns_false_on_connection_refused():
    """TCP refused -> False (no module list call possible)."""
    with patch(
        "socket.create_connection",
        side_effect=ConnectionRefusedError("nope"),
    ):
        result = _module_list("127.0.0.1", 1, 0.1)
    assert result is False


# --- probe_strict integration tests -----------------------------------------


async def test_probe_strict_dead_on_redislite_zombie(supervisor):
    """PING +PONG but MODULE LIST missing falkordb -> probe_strict DEAD.

    This is the redislite-zombie false-positive scenario that motivated
    Phase 3.1. With the old probe() (PING-only), the daemon would
    report falkor_ok=true and skip the boot-time auto-start. With
    probe_strict(), MODULE LIST catches the missing falkordb module
    and returns DEAD, triggering the auto-start in lifespan.py.
    """

    def fake_ping(host, port, timeout):
        return ProbeState.ALIVE  # PING fooled by redislite

    def fake_module(host, port, timeout):
        return False  # but no falkordb module loaded

    with patch(
        "janus_graph.daemon.falkordb_supervisor._redis_ping",
        side_effect=fake_ping,
    ), patch(
        "janus_graph.daemon.falkordb_supervisor._module_list",
        side_effect=fake_module,
    ):
        result = await supervisor.probe_strict()
    assert result is ProbeState.DEAD


async def test_probe_strict_alive_when_falkordb_module_loaded(supervisor):
    """PING +PONG AND MODULE LIST contains falkordb -> probe_strict ALIVE.

    Happy path: real FalkorDB boot. Both checks pass, supervisor reports ALIVE
    and lifespan.py skips the auto-start branch.
    """

    def fake_ping(host, port, timeout):
        return ProbeState.ALIVE

    def fake_module(host, port, timeout):
        return True

    with patch(
        "janus_graph.daemon.falkordb_supervisor._redis_ping",
        side_effect=fake_ping,
    ), patch(
        "janus_graph.daemon.falkordb_supervisor._module_list",
        side_effect=fake_module,
    ):
        result = await supervisor.probe_strict()
    assert result is ProbeState.ALIVE


async def test_probe_strict_short_circuits_on_ping_dead(supervisor):
    """PING DEAD -> probe_strict DEAD without calling MODULE LIST.

    Optimization: if PING is already dead, no point asking for MODULE LIST
    (and it would fail anyway with connection refused). The redislite-zombie
    false-positive is a SUBTLE case where PING passes but MODULE LIST fails.
    """

    def fake_ping(host, port, timeout):
        return ProbeState.DEAD

    module_called = False

    def fake_module(host, port, timeout):
        nonlocal module_called
        module_called = True
        return False

    with patch(
        "janus_graph.daemon.falkordb_supervisor._redis_ping",
        side_effect=fake_ping,
    ), patch(
        "janus_graph.daemon.falkordb_supervisor._module_list",
        side_effect=fake_module,
    ):
        result = await supervisor.probe_strict()
    assert result is ProbeState.DEAD
    assert module_called is False, "MODULE LIST should not be called when PING is DEAD"


async def test_probe_strict_warming_up_preserved(supervisor):
    """PING WARMING_UP -> probe_strict WARMING_UP; no MODULE LIST call.

    Redis hydrating from disk. MODULE LIST would still work but we do not want
    to fail boot just because the dataset is still loading.
    """

    def fake_ping(host, port, timeout):
        return ProbeState.WARMING_UP

    module_called = False

    def fake_module(host, port, timeout):
        nonlocal module_called
        module_called = True
        return False

    with patch(
        "janus_graph.daemon.falkordb_supervisor._redis_ping",
        side_effect=fake_ping,
    ), patch(
        "janus_graph.daemon.falkordb_supervisor._module_list",
        side_effect=fake_module,
    ):
        result = await supervisor.probe_strict()
    assert result is ProbeState.WARMING_UP
    assert module_called is False, "MODULE LIST should not be called when PING is WARMING_UP"
