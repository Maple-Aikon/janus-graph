"""FalkorDB supervisor with circuit-breaker policy.

Wraps (does NOT replace) ``FalkorDBServerManager`` so all Falkor lifecycle
stays in one place. Adds bounded-retry guard via ``FalkorCircuitBreaker``.

Per Plan #2 §3.3 Option C (Lazy Attach): probe ``host:port`` first via Redis
PING, fall back to pidfile check. This handles two cases:

  1. Falkor started by ai_suite_start.sh BEFORE daemon boots → pidfile path
     mismatch (e.g. legacy worktree path) but Redis is alive → attach.
  2. Falkor started by ``FalkorDBServerManager.start()`` → pidfile matches
     current cwd → both probes succeed.

Async surface:
  - is_running()  — true iff breaker.allow_probe AND supervisor-level probe
                    succeeds (Redis PING first, then pidfile fallback)
  - probe()       — single Falkor probe gated by breaker; updates breaker
                    on result. Records failure on timeout/connection-refused.
  - start()       — one-shot start attempt (only if breaker permits)

The supervisor is the single chokepoint that all /health handlers go through,
so the circuit state visible to /health reflects real probe outcomes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ..config import EngineConfig, FalkorCircuitSettings
from ..engine.server import FalkorDBServerManager
from .circuit_breaker import FalkorCircuitBreaker, CircuitSnapshot

logger = logging.getLogger("janus_graph.daemon.falkordb_supervisor")

# Probe timeout: short enough to avoid blocking /health, long enough to tolerate
# systemd-style daemon pauses. Tuned via FalkorCircuitSettings.probe_timeout_sec.
_DEFAULT_PROBE_TIMEOUT_SEC = 1.0


def _redis_ping(host: str, port: int, timeout: float) -> bool:
    """Sync Redis PING with timeout. Returns True iff server replied PONG.

    Uses stdlib socket to avoid forcing ``redis`` dep at this layer.
    """
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.sendall(b"PING" + bytes([10]))
            data = s.recv(64)
            return b"PONG" in data or b"+PONG" in data
    except (OSError, TimeoutError):
        return False


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
        manager: Optional[FalkorDBServerManager] = None,
    ) -> None:
        self._engine = engine_config or EngineConfig()
        self._circuit_cfg = circuit_config or FalkorCircuitSettings()
        self._manager = manager or FalkorDBServerManager(self._engine)
        self._breaker = FalkorCircuitBreaker(self._circuit_cfg)

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
            logger.debug("falkordb_supervisor: probe skipped (circuit OPEN)")
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


__all__ = ["DaemonSupervisor"]
