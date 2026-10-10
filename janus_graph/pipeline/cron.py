"""Main periodic queue batch sweeper and reaper with graceful shutdown."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from ..config import JanusSettings, PipelineConfig, load_config, resolve_home_relative
from ..core.contracts import Settings
from ..report.dispatcher import ReportDispatcher
from ..report.models import ReportSeverity
from .queue import EpisodeQueue, EpisodeRecord
from .worker import EpisodeWorker

logger = logging.getLogger("janus_graph.pipeline.cron")

PROCESSING_TIMEOUT_SECONDS = int(os.environ.get("GRAPHITI_PROCESSING_TIMEOUT", "900"))
WORKER_CONCURRENCY = int(os.environ.get("GRAPHITI_WORKER_CONCURRENCY", "2"))
REAPER_LIMIT = int(os.environ.get("GRAPHITI_REAPER_LIMIT", "500"))
SWEEP_TIMEOUT_SECONDS = float(os.environ.get("GRAPHITI_SWEEP_TIMEOUT", "900.0"))
PER_RECORD_TIMEOUT_SECONDS = float(os.environ.get("GRAPHITI_PER_RECORD_TIMEOUT", "900.0"))


async def run_cron_sweep(
    settings: Optional[Union[JanusSettings, Settings]] = None,
    batch_size: Optional[int] = None,
    concurrency: int = WORKER_CONCURRENCY,
    sweep_timeout_sec: float = SWEEP_TIMEOUT_SECONDS,
    record_timeout_sec: float = PER_RECORD_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Execute a full reaper + worker sweep over pending queue records."""
    cfg = settings or load_config()
    # contracts.Settings carries `pipeline: PipelineSettings`, whose field set
    # is {queue, dream, heuristics_active} -- it has NO drain_batch_size,
    # worker_concurrency, attempt_timeout_sec or queue_db_path. Measured
    # 2026-10-10: with a contracts.Settings and batch_size=None this used to
    # raise AttributeError at cfg.pipeline.drain_batch_size.
    # contracts.Settings is still accepted by the signature (tests build one),
    # so read those four through getattr instead of a direct attribute chain.
    # NOTE: `or DEFAULT` is NOT a safe rewrite -- a legal value of 0 would be
    # swallowed into the default (attempt_timeout_sec=0 means "no timeout").
    pipeline_cfg = getattr(cfg, "pipeline", None)
    actual_batch_size = (
        batch_size
        if (batch_size is not None and batch_size > 0)
        else getattr(pipeline_cfg, "drain_batch_size", None)
    )
    if actual_batch_size is None:
        # Read the declared default instead of hardcoding a number here: the
        # class default is 50 but the live config.yaml pins 20, and any literal
        # in this function would silently drift from both.
        actual_batch_size = PipelineConfig.model_fields["drain_batch_size"].default
    actual_concurrency = (
        getattr(pipeline_cfg, "worker_concurrency", None)
        if concurrency == WORKER_CONCURRENCY
        else concurrency
    ) or WORKER_CONCURRENCY
    # NB the trailing `or WORKER_CONCURRENCY` is kept EXACTLY as it was, and it
    # still spans both ternary branches. Replacing it with an `is None` test
    # would let an explicit concurrency=0 through to asyncio.Semaphore(0),
    # which never releases -- a behaviour change dressed as a cleanup.
    attempt_timeout = getattr(pipeline_cfg, "attempt_timeout_sec", None)
    actual_record_timeout = (
        float(attempt_timeout)
        if (record_timeout_sec == PER_RECORD_TIMEOUT_SECONDS and attempt_timeout is not None)
        else record_timeout_sec
    )
    actual_sweep_timeout = (
        float(attempt_timeout)
        if (sweep_timeout_sec == SWEEP_TIMEOUT_SECONDS and attempt_timeout is not None)
        else sweep_timeout_sec
    )
    actual_reap_timeout = (
        int(attempt_timeout) if attempt_timeout is not None else PROCESSING_TIMEOUT_SECONDS
    )
    # Same hasattr guard as dream.py: contracts.Settings keeps the path
    # under ``paths``. Measured 2026-10-02: a bare contracts.Settings raises
    # AttributeError on ``pipeline.drain_batch_size`` at the line ABOVE this
    # one, so that path is unreachable *today* -- this is defence-in-depth for
    # the day contracts.PipelineSettings grows drain_batch_size, and the
    # branch must not become cwd-relative when it does.
    #
    # `cfg.paths` must be read through getattr on the SETTINGS too: a
    # JanusSettings has no ``paths`` at all, and `getattr(cfg.paths, ...)`
    # evaluates ``cfg.paths`` eagerly, so the old spelling raised
    # AttributeError here instead of falling back. Measured 2026-10-10 with a
    # live JanusSettings: the AttributeError fires the moment the branch is
    # taken, i.e. the "defence" would have been the crash.
    pipeline_path = getattr(pipeline_cfg, "queue_db_path", None)
    if pipeline_path is None:
        paths_cfg = getattr(cfg, "paths", None)
        pipeline_path = getattr(paths_cfg, "queue_db_path", None)
    if pipeline_path is None:
        pipeline_path = "./data/queue.db"
    db_path = resolve_home_relative(str(pipeline_path))
    queue = EpisodeQueue(db_path)
    worker = EpisodeWorker(queue, cfg)
    dispatcher = ReportDispatcher.from_settings(cfg)

    start_time = time.monotonic()

    # Phase 1: Reaper-first (unlock stuck processing rows)
    reaped_count = await queue.reap_stuck_processing(timeout_sec=actual_reap_timeout)
    if reaped_count > 0:
        logger.info("Reaped %d stuck processing records.", reaped_count)

    # Phase 2: Claim batch of queued records
    records = await queue.claim_next_batch(limit=actual_batch_size)
    processed_count = len(records)
    succeeded_count = 0
    failed_count = 0

    if records:
        logger.info(
            "Claimed %d records for processing (concurrency=%d)",
            processed_count,
            actual_concurrency,
        )
        sem = asyncio.Semaphore(actual_concurrency)

        async def _safe_process(rec: EpisodeRecord):
            nonlocal succeeded_count, failed_count
            # Check soft sweep deadline
            if time.monotonic() - start_time > actual_sweep_timeout:
                logger.warning("Sweep soft timeout reached; deferring record %s", rec.id)
                await queue.mark_failed(rec.id, "Sweep soft timeout exceeded")
                failed_count += 1
                return

            async with sem:
                ok = await worker.process_record(rec, timeout_sec=actual_record_timeout)
                if ok:
                    succeeded_count += 1
                else:
                    failed_count += 1

        await asyncio.gather(*(_safe_process(rec) for rec in records), return_exceptions=True)

    duration_ms = round((time.monotonic() - start_time) * 1000, 2)
    stats_final = queue.get_stats()

    summary = {
        "processed": processed_count,
        "succeeded": succeeded_count,
        "failed": failed_count,
        "reaped": reaped_count,
        "duration_ms": duration_ms,
        "queued_remaining": stats_final.get("queued", 0),
        "dlq_count": stats_final.get("dlq", 0),
        "total_episodes": queue.count_total(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Dispatch consolidated sweep summary
    try:
        severity = ReportSeverity.ERROR if failed_count > 0 else ReportSeverity.INFO
        await asyncio.shield(
            dispatcher.emit_quick(
                kind="cron_sweep",
                summary=f"Sweep completed: {succeeded_count}/{processed_count} ok, {failed_count} fail, {stats_final.get('queued', 0)} left ({duration_ms}ms)",
                severity=severity,
                details=summary,
            )
        )
    except Exception as err:
        logger.warning("Failed to emit sweep report: %s", err)

    return summary
