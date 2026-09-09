"""FalkorDB supervisor with circuit-breaker policy + bounded-restart hook.

Wraps (does NOT replace) ``FalkorDBServerManager`` so all Falkor lifecycle
stays in one place. Adds:
  - ``FalkorCircuitBreaker`` — 3-state probe gating
  - ``FalkorRestartPolicy`` — bounded-rate restart on CLOSED → OPEN (Phase 1)

Per Plan #2 §3.3 Option C (Lazy Attach): probe ``host:port`` first via Redis
PING, fall back to pidfile check. This handles two cases:

  1. Falkor started by ai_suite_start.sh BEFORE daemon boots → pidfile path
     mismatch (e.g. legacy worktree path) but Redis is alive → attach.
  2. Falkor started by ``FalkorDBServerManager.start()`` → pidfile matches
     current cwd → both probes succeed.

Phase 1 probe semantics (Plan #3 §9 HIGH #1):
  - ``ALIVE``: PING returned ``+PONG``
  - ``WARMING_UP``: Redis bound port but responded ``-LOADING``; do NOT
    count as failure and do NOT trip the circuit-breaker.
  - ``DEAD``: connection refused / timeout / malformed reply

Async surface:
  - is_running()  — true iff breaker.allow_probe AND supervisor-level probe
                    succeeds (Redis PING first, then pidfile fallback)
  - probe()       — single Falkor probe gated by breaker; updates breaker
                    on result. Records failure on DEAD only.
  - start()       — one-shot start attempt (only if breaker permits)

The supervisor is the single chokepoint that all /health handlers go through,
so the circuit state visible to /health reflects real probe outcomes.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Optional

from ..config import (
    EngineConfig,
    FalkorCircuitSettings,
    FalkorRestartPolicySettings,
)
from ..engine.server import FalkorDBServerManager
from .circuit_breaker import FalkorCircuitBreaker, CircuitSnapshot
from .falkordb_restart_policy import FalkorRestartPolicy

logger = logging.getLogger("janus_graph.daemon.falkordb_supervisor")


class ProbeState(str, Enum):
    """FalkorDB liveness classification returned by ``DaemonSupervisor.probe_state``.

    Distinct from ``CircuitState`` (which is a probe *gating* state):
    ``ProbeState`` describes the *current* engine observation, while
    ``CircuitState`` describes the policy decision to probe at all.
    """

    ALIVE = "alive"               # PING → +PONG
    WARMING_UP = "warming_up"     # PING → -LOADING  (Redis hydrating from disk)
    DEAD = "dead"                 # connection refused / timeout / parse fail

# Probe timeout: short enough to avoid blocking /health, long enough to tolerate
# systemd-style daemon pauses. Tuned via FalkorCircuitSettings.probe_timeout_sec.
_DEFAULT_PROBE_TIMEOUT_SEC = 1.0


def _redis_ping(host: str, port: int, timeout: float) -> ProbeState:
    """Sync Redis PING with timeout. Returns ProbeState classification.

    Uses stdlib socket to avoid forcing ``redis`` dep at this layer.

    RESP wire format (minimal):
      - ``+PONG\
\
``  → ALIVE
      - ``-LOADING ...``  → WARMING_UP (Redis hydrating from disk)
      - other / OSError / TimeoutError → DEAD
    """
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.sendall(b"PING" + bytes([13, 10]))
            data = s.recv(64)
    except (OSError, TimeoutError):
        return ProbeState.DEAD
    if b"+PONG" in data or b"PONG" in data:
        return ProbeState.ALIVE
    if b"-LOADING" in data:
        return ProbeState.WARMING_UP
    return ProbeState.DEAD


class DaemonSupervisor:
    """Falkor supervisor with circuit-breaker gating.

    Usage (Phase 2):
        supervisor = DaemonSupervisor(engine_cfg, circuit_cfg)
        await supervisor.probe()  # one-off /health check
        snap = supervisor.snapshot()
    """

    def __init__(
        self,
        engine_config: Optional[EngineConfig] = None,
        circuit_config: Optional[FalkorCircuitSettings] = None,
        restart_policy_config: Optional[FalkorRestartPolicySettings] = None,
        manager: Optional[FalkorDBServerManager] = None,
    ) -> None:
        self._engine = engine_config or EngineConfig()
        self._circuit_cfg = circuit_config or FalkorCircuitSettings()
        self._restart_policy_cfg = restart_policy_config or FalkorRestartPolicySettings()
        self._manager = manager or FalkorDBServerManager(self._engine)
        self._breaker = FalkorCircuitBreaker(self._circuit_cfg)
        # Phase 1 (Plan #3): bounded restart policy bound to breaker
        # CLOSED→OPEN hook. The policy owns the rate-limiter; the breaker
        # fires the callback exactly once per CLOSED→OPEN transition.
        self._restart_policy = FalkorRestartPolicy(
            settings=self._restart_policy_cfg,
            manager=self._manager,
        )
        self._breaker.on_open(self._on_breaker_opened)

    def _on_breaker_opened(self, snap: CircuitSnapshot, _manager_sentinel) -> None:
        """Internal shim: route breaker hook → policy.on_circuit_opened.

        The signature matches circuit_breaker.OnCircuitOpenedCallback. We
        pass ``self._manager`` directly so the policy doesn't need to rely
        on the sentinel binding path.
        """
        self._restart_policy.on_circuit_opened(snap, self._manager)

    # ─── public ──────────────────────────────────────────────────────────

    async def _probe_redis(self) -> bool:
        """Plan §3.3 Option C: probe host:port via Redis PING.

        Returns True iff server responded PONG within timeout.
        Runs in to_thread to keep async loop responsive.
        """
        timeout = float(
            getattr(self._circuit_cfg, "probe_timeout_sec", _DEFAULT_PROBE_TIMEOUT_SEC)
        )
        return await asyncio.to_thread(
            _redis_ping, self._engine.host, int(self._engine.port), timeout
        )

    async def is_running(self) -> bool:
        """True iff breaker permits probe AND either Redis PING succeeds
        OR manager pidfile reports alive.
        """
        if not await self._breaker.allow_probe():
            return False
        # Try Redis PING first (Plan §3.3 Option C — handles pidfile path mismatch).
        if await self._probe_redis():
            return True
        # Fallback: pidfile check (covers edge case where Redis-protocol probe
        # races with start-up; pidfile is authoritative for "we started it").
        return await asyncio.to_thread(self._manager.is_running)

    async def probe(self) -> bool:
        """Single Falkor probe — gated by circuit-breaker.

        Returns True iff Falkor is alive (Redis PING or pidfile).
        Updates breaker on result (success → reset, failure → record_failure).
        Does NOT call ``start()`` — that's a separate explicit operation.
        """
        if not await self._breaker.allow_probe():
            # OPEN-skip: no syscall, no Falkor touch (T8 verification).
            logger.info("falkordb_supervisor: probe skipped (circuit OPEN)")
            return False

        try:
            # Plan §3.3 Option C: Redis PING first.
            alive = await self._probe_redis()
            if not alive:
                # Fallback to pidfile (handles edge case where Redis is mid-startup).
                alive = await asyncio.to_thread(self._manager.is_running)
        except Exception as e:
            # Defensive: any probe exception counts as failure.
            logger.warning("falkordb_supervisor: probe raised %s", e)
            alive = False

        if alive:
            await self._breaker.record_success()
        else:
            await self._breaker.record_failure()
        return alive

    async def start(self) -> bool:
        """Attempt Falkor start (only if breaker permits).

        Returns True iff manager.start() succeeded.
        Does NOT record breaker success — that requires a follow-up probe.
        """
        if not await self._breaker.allow_probe():
            logger.warning("falkordb_supervisor: start skipped (circuit OPEN)")
            return False
        try:
            ok = await asyncio.to_thread(self._manager.start)
            return ok
        except Exception as e:
            logger.error("falkordb_supervisor: start raised %s", e)
            await self._breaker.record_failure()
            return False

    async def stop(self) -> bool:
        """Stop Falkor (no breaker check — operator action)."""
        try:
            return await asyncio.to_thread(self._manager.stop)
        except Exception as e:
            logger.error("falkordb_supervisor: stop raised %s", e)
            return False

    def snapshot(self) -> CircuitSnapshot:
        """Read-only breaker snapshot."""
        return self._breaker.snapshot()

    # ─── introspection ───────────────────────────────────────────────────

    @property
    def breaker(self) -> FalkorCircuitBreaker:
        """Direct breaker access (for tests; do not mutate)."""
        return self._breaker

    @property
    def manager(self) -> FalkorDBServerManager:
        """Direct manager access (for tests; do not mutate)."""
        return self._manager

    @property
    def restart_policy(self) -> FalkorRestartPolicy:
        """Direct restart-policy access (for /health wiring + tests)."""
        return self._restart_policy


__all__ = ["DaemonSupervisor"]
