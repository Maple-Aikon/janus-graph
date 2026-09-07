"""Daemon lifespan orchestrator.

Phase 2 boot sequence (per Plan #2 §4.1a):
  1. parse config.yaml → JanusSettings
  2. attach_or_start_falkordb via supervisor (probe + lazy start)
  3. init EpisodeQueue (Phase 2: stub — Phase 3 wires real queue)
  4. warm graphiti client (Phase 2: lazy — Phase 3 instantiates)
  5. spawn aiohttp HTTP server on 127.0.0.1:cfg.daemon.http.port
  6. log "daemon ready on port N (cron disabled — Phase 4)"

Graceful shutdown:
  - stop accepting new HTTP requests
  - await in-flight handlers (aiohttp default)
  - close supervisor (no-op for sync manager)
  - exit 0
"""

from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass
from typing import Optional

from aiohttp import web

from ..config import HTTPSettings, JanusSettings
from .falkordb_supervisor import DaemonSupervisor
from .http_server import __version__, build_app

logger = logging.getLogger("janus_graph.daemon.lifespan")


@dataclass
class DaemonContext:
    """Shared context held by all HTTP handlers."""

    settings: JanusSettings
    supervisor: DaemonSupervisor
    http_settings: HTTPSettings
    version: str
    shutdown_event: asyncio.Event
    runner: Optional[web.AppRunner] = None
    site: Optional[web.BaseSite] = None

    def request_shutdown(self) -> None:
        """Signal main loop to exit (called by /shutdown handler)."""
        self.shutdown_event.set()


async def boot(ctx: DaemonContext) -> None:
    """Phase 2 boot: probe Falkor → warm graphiti stub → start HTTP.

    Per Plan #2 §4.1a, this is the full boot flow minus cron.
    """
    logger.info("daemon: booting (version=%s)", ctx.version)

    # 1. Attach/start Falkor via supervisor (lazy attach — Option C from plan).
    falkor_ok = await ctx.supervisor.probe()
    if not falkor_ok:
        logger.warning(
            "daemon: Falkor not alive — attempting start (circuit permits)"
        )
        started = await ctx.supervisor.start()
        if started:
            falkor_ok = await ctx.supervisor.probe()
            logger.info("daemon: Falkor started OK: alive=%s", falkor_ok)
        else:
            logger.warning(
                "daemon: Falkor start failed — entering degraded mode "
                "(/health will return 503 until recoverable)"
            )
    else:
        logger.info("daemon: Falkor alive on first probe")

    # 2. Phase 3 placeholder for EpisodeQueue.init() — skipped in Phase 2.
    logger.info("daemon: EpisodeQueue init deferred to Phase 3 (Phase 2 PR scope)")

    # 3. Phase 3 placeholder for graphiti warm — skipped in Phase 2.
    logger.info("daemon: graphiti client warm deferred to Phase 3 (Phase 2 PR scope)")

    # 4. Build + start aiohttp.
    app = build_app(ctx)
    ctx.runner = web.AppRunner(app)
    await ctx.runner.setup()
    ctx.site = web.TCPSite(
        ctx.runner,
        host=ctx.http_settings.host,
        port=ctx.http_settings.port,
    )
    await ctx.site.start()
    logger.info(
        "daemon: ready on port %d (cron disabled — Phase 4) "
        "— http://%s:%d/health",
        ctx.http_settings.port,
        ctx.http_settings.host,
        ctx.http_settings.port,
    )


async def shutdown(ctx: DaemonContext) -> None:
    """Graceful shutdown — max 30s (matches Plan #2 §4.1a contract).

    Order:
      1. stop accepting new HTTP requests (site.stop)
      2. await in-flight handlers (runner.cleanup)
      3. close supervisor (no-op for sync Falkor manager)
    """
    logger.info("daemon: shutting down")
    if ctx.site is not None:
        try:
            await asyncio.wait_for(ctx.site.stop(), timeout=10)
        except asyncio.TimeoutError:
            logger.warning("daemon: site.stop timed out after 10s")
    if ctx.runner is not None:
        try:
            await asyncio.wait_for(ctx.runner.cleanup(), timeout=20)
        except asyncio.TimeoutError:
            logger.warning("daemon: runner.cleanup timed out after 20s")
    logger.info("daemon: shutdown complete (exit 0)")


def install_signal_handlers(loop: asyncio.AbstractEventLoop, ctx: DaemonContext) -> None:
    """Wire SIGTERM/SIGINT to graceful shutdown event."""

    def _trigger() -> None:
        logger.info("daemon: signal received — graceful shutdown")
        ctx.request_shutdown()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _trigger)
        except NotImplementedError:
            # Windows / non-Unix — fall back to default handlers.
            pass


async def run(settings: JanusSettings) -> int:
    """Entry point — boot → wait → shutdown.

    Returns 0 on clean exit.
    """
    supervisor = DaemonSupervisor(
        engine_config=settings.engine,
        circuit_config=settings.daemon.falkor_circuit,
    )
    ctx = DaemonContext(
        settings=settings,
        supervisor=supervisor,
        http_settings=settings.daemon.http,
        version=__version__,
        shutdown_event=asyncio.Event(),
    )

    loop = asyncio.get_running_loop()
    install_signal_handlers(loop, ctx)

    try:
        await boot(ctx)
        await ctx.shutdown_event.wait()
    finally:
        await shutdown(ctx)

    return 0


__all__ = [
    "DaemonContext",
    "boot",
    "shutdown",
    "install_signal_handlers",
    "run",
]
