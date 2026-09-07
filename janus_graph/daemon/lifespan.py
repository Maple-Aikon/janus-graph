"""Daemon lifespan orchestrator.

Phase 2 boot sequence (per Plan #2 §4.1a):
  1. parse config.yaml → JanusSettings
  2. attach_or_start_falkordb via supervisor (probe + lazy start)
  3. init EpisodeQueue + EpisodeQueueAdapter (Phase 3 — wired)
  4. warm graphiti stub (Phase 3 — SearchEngine wired, Graphiti itself
     stays lazy until a caller asks for it)
  5. spawn aiohttp HTTP server on 127.0.0.1:cfg.daemon.http.port
  6. log "daemon ready on port N (cron disabled — Phase 4)"

Phase 3 additions:
  - EpisodeQueue + EpisodeQueueAdapter initialized on boot (idempotent
    schema migration in to_thread).
  - EmbeddingClient opened on boot (aiohttp session).
  - SearchEngine constructed (lazy redis connect — fail-soft).

Phase 4 additions:
  - CronLoop wired when daemon.cron_enabled=true (gated).
    Sweep + dream run in-process; ai_cronjob.sh janus subshells SKIP
    via /health alive-check. See janus_graph/daemon/cron_loop.py.

Graceful shutdown:
  - stop CronLoop tasks (Phase 4 — cancel + await drain)
  - stop accepting new HTTP requests
  - await in-flight handlers (aiohttp default)
  - close supervisor (no-op for sync manager)
  - close SearchEngine + EmbeddingClient
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
from ..pipeline.queue import EpisodeQueue
from .cron_loop import CronLoop
from .embedding_client import EmbeddingClient
from .episode_queue_adapter import EpisodeQueueAdapter
from .falkordb_supervisor import DaemonSupervisor
from .http_server import __version__, build_app
from .search_engine import SearchEngine

logger = logging.getLogger("janus_graph.daemon.lifespan")


@dataclass
class DaemonContext:
    """Shared context held by all HTTP handlers.

    Phase 2 fields: settings, supervisor, http_settings, version,
                    shutdown_event, runner, site.
    Phase 3 fields: queue (EpisodeQueue), queue_adapter (EpisodeQueueAdapter),
                    embedding_client (EmbeddingClient),
                    search_engine (SearchEngine).
    Phase 4 fields: cron_loop (CronLoop) — only populated when cron_enabled.
    """

    settings: JanusSettings
    supervisor: DaemonSupervisor
    http_settings: HTTPSettings
    version: str
    shutdown_event: asyncio.Event
    queue: Optional[EpisodeQueue] = None
    queue_adapter: Optional[EpisodeQueueAdapter] = None
    embedding_client: Optional[EmbeddingClient] = None
    search_engine: Optional[SearchEngine] = None
    cron_loop: Optional["CronLoop"] = None
    runner: Optional[web.AppRunner] = None
    site: Optional[web.BaseSite] = None

    def request_shutdown(self) -> None:
        """Signal main loop to exit (called by /shutdown handler)."""
        self.shutdown_event.set()


async def _build_phase3_components(ctx: DaemonContext) -> None:
    """Phase 3 wiring: queue + adapter + embedding + search.

    Each step is independently fail-soft — Falkor-down leaves
    ctx.search_engine constructed but unconnected (search will return
    FALKOR_DOWN at request time). SQLite errors propagate.
    """
    queue_path = ctx.settings.pipeline.queue_db_path
    ctx.queue = EpisodeQueue(db_path=queue_path)
    ctx.queue_adapter = EpisodeQueueAdapter(
        queue=ctx.queue,
        lock_settings=ctx.settings.daemon.lock,
    )
    await ctx.queue_adapter.start()
    logger.info("daemon: EpisodeQueueAdapter ready (db=%s, mode=%s)",
                queue_path, ctx.settings.daemon.lock.mode)

    ctx.embedding_client = EmbeddingClient(ctx.settings.daemon.search_graph)
    await ctx.embedding_client.start()
    logger.info("daemon: EmbeddingClient ready (url=%s, conc=%d)",
                ctx.settings.daemon.search_graph.embedding_base_url,
                ctx.settings.daemon.search_graph.embedding_concurrency)

    ctx.search_engine = SearchEngine(
        engine_config=ctx.settings.engine,
        search_settings=ctx.settings.daemon.search_graph,
        embedding_client=ctx.embedding_client,
    )
    await ctx.search_engine.start()
    if ctx.search_engine._redis is not None:
        logger.info("daemon: SearchEngine ready (redis connected)")
    else:
        logger.warning("daemon: SearchEngine constructed but Falkor unreachable — "
                       "/search/graph will return 503 until Falkor recovers")


async def _shutdown_phase3_components(ctx: DaemonContext) -> None:
    """Reverse the Phase 3 init order."""
    if ctx.search_engine is not None:
        await ctx.search_engine.close()
    if ctx.embedding_client is not None:
        await ctx.embedding_client.stop()
    # queue.close() intentionally omitted — EpisodeQueue holds the WAL
    # connection; we let the process exit close it via GC. Phase 4 cron
    # may share this queue and shouldn't see early-close side effects.


async def boot(ctx: DaemonContext) -> None:
    """Phase 2+3+4 boot: probe Falkor → wire Phase 3 → start HTTP → start cron.

    Per Plan #2 §4.1a + §6 Phase 4, full boot flow including cron_loop
    (gated by daemon.cron_enabled).
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

    # 2. Phase 3 wiring (queue + embedding + search).
    await _build_phase3_components(ctx)

    # 3. Phase 4 cron loop — gated by daemon.cron_enabled. ai_cronjob.sh
    #    alive-check skips janus subshells when this daemon is up.
    ctx.cron_loop = CronLoop(
        settings=ctx.settings,
        circuit_state_provider=lambda: ctx.supervisor.breaker.state,
    )
    await ctx.cron_loop.start()

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
    cron_state = "ENABLED — Phase 4" if ctx.cron_loop.running else "disabled — Phase 4 gate off"
    logger.info(
        "daemon: ready on port %d (cron %s) "
        "— http://%s:%d/health",
        ctx.http_settings.port,
        cron_state,
        ctx.http_settings.host,
        ctx.http_settings.port,
    )


async def run(settings: JanusSettings) -> int:
    """Run daemon until SIGTERM/SIGINT or /shutdown trigger."""
    ctx = DaemonContext(
        settings=settings,
        supervisor=DaemonSupervisor(
            engine_config=settings.engine,
            circuit_config=settings.daemon.falkor_circuit,
        ),
        http_settings=settings.daemon.http,
        version=__version__,
        shutdown_event=asyncio.Event(),
    )

    # Signal handlers — graceful shutdown on SIGTERM/SIGINT.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, ctx.request_shutdown)

    try:
        await boot(ctx)
        await ctx.shutdown_event.wait()
    finally:
        logger.info("daemon: shutting down (graceful, max 30s)")
        # Phase 4: stop cron_loop FIRST so it doesn't start a new sweep
        # while we're tearing down the queue + supervisor.
        if ctx.cron_loop is not None:
            try:
                await asyncio.wait_for(ctx.cron_loop.stop(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("daemon: cron_loop stop timed out")
        # Phase 3 cleanup.
        try:
            await asyncio.wait_for(_shutdown_phase3_components(ctx), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("daemon: Phase 3 shutdown timed out")
        # HTTP server cleanup.
        if ctx.site is not None:
            try:
                await asyncio.wait_for(ctx.site.stop(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("daemon: site.stop timed out")
        if ctx.runner is not None:
            try:
                await asyncio.wait_for(ctx.runner.cleanup(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("daemon: runner.cleanup timed out")
    logger.info("daemon: shutdown complete (version=%s)", ctx.version)
    return 0
