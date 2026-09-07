"""Phase 4 daemon-internal cron loop.

Replaces the legacy ``ai_cronjob.sh`` janus sweep + dream subshells with an
in-process asyncio task that runs while the daemon is alive.

Gated by ``cfg.daemon.cron_enabled`` (default False):
  - False: CronLoop.start() returns immediately, no task spawned. Legacy
    ai_cronjob.sh keeps the sweep alive via ``uv run janus-graph sweep``.
  - True : CronLoop spawns 2 background tasks:
              * sweep_loop: ticks every ``cfg.pipeline.cron_interval_min`` min
                calling ``pipeline.cron.run_cron_sweep(settings)``
              * dream_loop: ticks every 60s checking for 02:30 ICT trigger;
                when matched, calls ``pipeline.dream.run_dream_consolidation``
              Both tasks share the daemon's Falkor circuit-breaker state and
              honor it: when OPEN, sweep tick is skipped (logged at warning),
              dream tick is skipped entirely.
  - Rollback: flip cron_enabled back to false → restart daemon → legacy
    ai_cronjob.sh resumes sweep work. No data loss (queue is SQLite-backed).

Per Plan #2 §4.1b + §6 Phase 4 verify gates V1-V5.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from ..config import JanusSettings
from ..pipeline.cron import run_cron_sweep
from ..pipeline.dream import run_dream_consolidation

logger = logging.getLogger("janus_graph.daemon.cron_loop")

# Map CircuitState enum → "CLOSED" / "OPEN" / "HALF_OPEN" string for the
# state_provider callable. The provider returns the string we compare
# against in the loop body. Using str() avoids importing the enum here
# (and the cron_loop module is intentionally cheap to import).
_CIRCUIT_OPEN = "OPEN"
_CIRCUIT_CLOSED = "CLOSED"
_CIRCUIT_HALF_OPEN = "HALF_OPEN"


class CronLoop:
    """Phase 4 in-process cron loop. Honors daemon.cron_enabled + Falkor circuit.

    Lifecycle:
      loop = CronLoop(settings, circuit_state_provider)
      await loop.start()     # spawns tasks if cron_enabled; no-op otherwise
      ...
      await loop.stop()      # cancels tasks + awaits graceful drain

    State (for /health observability):
      .running : bool
      .last_sweep_at : Optional[float]   # monotonic seconds
      .last_sweep_stats : Optional[dict]
      .last_dream_at : Optional[float]
      .last_dream_result : Optional[dict]
      .tick_count : int
    """

    DREAM_HOUR_ICT = 2     # 02:30 ICT per ai_cronjob.sh legacy
    DREAM_MINUTE_ICT = 30
    DREAM_CHECK_INTERVAL_SEC = 60  # check the wall clock once per minute
    # Sweep default 10 min matches cfg.pipeline.cron_interval_min. Tests
    # override cfg.pipeline.cron_interval_min to drive faster ticks.

    def __init__(
        self,
        settings: JanusSettings,
        circuit_state_provider,
    ) -> None:
        self._settings = settings
        self._circuit_state_provider = circuit_state_provider
        self._tasks: list[asyncio.Task] = []
        self._shutdown = asyncio.Event()
        # Observability
        self.running: bool = False
        self.last_sweep_at: Optional[float] = None
        self.last_sweep_stats: Optional[dict] = None
        self.last_dream_at: Optional[float] = None
        self.last_dream_result: Optional[dict] = None
        self.tick_count: int = 0
        self.skipped_circuit_open: int = 0

    # ─── lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Spawn sweep + dream tasks if cron_enabled; otherwise no-op."""
        if not self._settings.daemon.cron_enabled:
            logger.info(
                "cron_loop: daemon.cron_enabled=false → CronLoop NOT spawned "
                "(legacy ai_cronjob.sh keeps sweep alive)"
            )
            return
        if self.running:
            logger.warning("cron_loop: start() called while already running")
            return

        self._shutdown.clear()
        self.running = True
        interval_min = self._settings.pipeline.cron_interval_min
        self._tasks = [
            asyncio.create_task(self._sweep_loop(interval_min), name="cron-sweep"),
            asyncio.create_task(self._dream_loop(), name="cron-dream"),
        ]
        logger.info(
            "cron_loop: started (sweep_every=%dmin, dream_at=%02d:%02d ICT)",
            interval_min, self.DREAM_HOUR_ICT, self.DREAM_MINUTE_ICT,
        )

    async def stop(self) -> None:
        """Cancel tasks; await graceful drain (max 30s)."""
        if not self.running:
            return
        self._shutdown.set()
        for t in self._tasks:
            t.cancel()
        # Wait for graceful drain.
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("cron_loop: stop() timed out after 30s")
        self._tasks = []
        self.running = False
        logger.info("cron_loop: stopped (tick_count=%d)", self.tick_count)

    # ─── sweep loop ─────────────────────────────────────────────────────

    async def _sweep_loop(self, interval_min: int) -> None:
        """Run run_cron_sweep every ``interval_min`` minutes.

        Skips tick when Falkor circuit is OPEN (per Plan #2 §4.1b):
        the supervisor will already be probing; double-work wastes the
        embedding budget. Skip is logged at WARNING so operators see it.
        """
        interval_sec = max(1, int(interval_min * 60))
        # First tick: small delay so daemon can settle (Falkor probe finishes).
        try:
            await asyncio.wait_for(self._shutdown.wait(), timeout=5.0)
            return  # shutdown during initial delay
        except asyncio.TimeoutError:
            pass

        while not self._shutdown.is_set():
            # Honor Falkor circuit OPEN → skip.
            state = _state_str(self._circuit_state_provider())
            if state == _CIRCUIT_OPEN:
                self.skipped_circuit_open += 1
                logger.warning(
                    "cron_loop: sweep tick SKIPPED — Falkor circuit OPEN "
                    "(skip_count=%d)",
                    self.skipped_circuit_open,
                )
            else:
                try:
                    stats = await run_cron_sweep(self._settings)
                    self.last_sweep_at = asyncio.get_event_loop().time()
                    self.last_sweep_stats = stats
                    self.tick_count += 1
                    logger.info(
                        "cron_loop: sweep tick #%d done — %s",
                        self.tick_count,
                        _short_stats(stats),
                    )
                except Exception as exc:  # sweep failure must not crash loop
                    logger.exception("cron_loop: sweep tick raised: %s", exc)

            # Sleep until next tick OR shutdown.
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=interval_sec)
                return  # shutdown signal
            except asyncio.TimeoutError:
                continue  # next tick

    # ─── dream loop ─────────────────────────────────────────────────────

    async def _dream_loop(self) -> None:
        """Run run_dream_consolidation once per day at 02:30 ICT.

        Triggers when the wall clock minute matches DREAM_HOUR:MINUTE ICT
        and we haven't already run today (compare date string).
        """
        last_run_date: Optional[str] = None
        while not self._shutdown.is_set():
            now_utc = datetime.now(timezone.utc)
            # Convert to ICT (UTC+7, no DST in Vietnam).
            ict_hour = (now_utc.hour + 7) % 24
            # Note: when hour wraps past midnight, the date string is the
            # UTC date+1; for dream-trigger purposes we only care about the
            # clock matching — date string is the UTC date, used to dedupe
            # within the same UTC day so we don't double-fire.
            utc_date = now_utc.strftime("%Y-%m-%d")
            if (
                ict_hour == self.DREAM_HOUR_ICT
                and now_utc.minute >= self.DREAM_MINUTE_ICT
                and last_run_date != utc_date
            ):
                # Honor Falkor circuit OPEN — skip entirely.
                state = _state_str(self._circuit_state_provider())
                if state == _CIRCUIT_OPEN:
                    logger.warning(
                        "cron_loop: dream run SKIPPED — Falkor circuit OPEN "
                        "(will retry next minute window)"
                    )
                else:
                    logger.info("cron_loop: dream run STARTED")
                    try:
                        result = await run_dream_consolidation(self._settings)
                        self.last_dream_at = asyncio.get_event_loop().time()
                        self.last_dream_result = result
                        last_run_date = utc_date
                        logger.info(
                            "cron_loop: dream run COMPLETED — %s",
                            _short_stats(result),
                        )
                    except Exception as exc:
                        logger.exception("cron_loop: dream run raised: %s", exc)

            # Wait 60s before next check.
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self.DREAM_CHECK_INTERVAL_SEC
                )
                return
            except asyncio.TimeoutError:
                continue

    # ─── observability ──────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """State snapshot for /health body extension (Phase 4 observability)."""
        return {
            "enabled": self._settings.daemon.cron_enabled,
            "running": self.running,
            "tick_count": self.tick_count,
            "skipped_circuit_open": self.skipped_circuit_open,
            "last_sweep_at": self.last_sweep_at,
            "last_sweep_stats": self.last_sweep_stats,
            "last_dream_at": self.last_dream_at,
            "last_dream_result": self.last_dream_result,
        }


def _short_stats(stats: Optional[dict]) -> str:
    """One-line summary of a sweep/dream result dict."""
    if not stats:
        return "no stats"
    parts = []
    for k in ("succeeded", "succeeded_count", "failed", "failed_count",
              "processed", "phase_2_deduplication", "phase_3_orphan_pruning",
              "phase_4_dlq_repair", "duration_ms"):
        if k in stats:
            parts.append(f"{k}={stats[k]}")
    return " ".join(parts) if parts else str(stats)[:200]


def _state_str(raw) -> str:
    """Coerce a Falkor circuit-state value into our 3-string vocabulary.

    Accepts:
      - CircuitState enum (has .name / str())
      - plain str ("CLOSED" / "OPEN" / "HALF_OPEN")
      - anything else → unknown, treated as not-OPEN (allow tick).
    """
    if raw is None:
        return _CIRCUIT_CLOSED
    name = getattr(raw, "name", None)
    if name:
        return name
    s = str(raw).upper()
    if s in (_CIRCUIT_CLOSED, _CIRCUIT_OPEN, _CIRCUIT_HALF_OPEN):
        return s
    return _CIRCUIT_CLOSED  # unknown → be permissive (don't skip on guess)
