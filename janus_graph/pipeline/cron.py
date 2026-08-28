"""Main periodic queue batch sweeper and reaper with graceful shutdown."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union
from ..config import JanusSettings, load_config
from ..core.contracts import Settings
from ..report.dispatcher import ReportDispatcher
from ..report.models import ReportSeverity
from .queue import EpisodeQueue, EpisodeRecord
from .worker import EpisodeWorker

logger = logging.getLogger("janus_graph.pipeline.cron")

PROCESSING_TIMEOUT_SECONDS = int(os.environ.get("GRAPHITI_PROCESSING_TIMEOUT", "300"))
WORKER_CONCURRENCY = int(os.environ.get("GRAPHITI_WORKER_CONCURRENCY", "3"))
SWEEP_LIMIT = int(os.environ.get("GRAPHITI_SWEEP_LIMIT", "6"))
REAPER_LIMIT = int(os.environ.get("GRAPHITI_REAPER_LIMIT", "500"))
SWEEP_TIMEOUT_SECONDS = float(os.environ.get("GRAPHITI_SWEEP_TIMEOUT", "420.0"))
PER_RECORD_TIMEOUT_SECONDS = float(os.environ.get("GRAPHITI_PER_RECORD_TIMEOUT", "300.0"))


async def run_cron_sweep(
    settings: Optional[Union[JanusSettings, Settings]] = None,
    batch_size: int = SWEEP_LIMIT,
    concurrency: int = WORKER_CONCURRENCY,
    sweep_timeout_sec: float = SWEEP_TIMEOUT_SECONDS,
    record_timeout_sec: float = PER_RECORD_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Execute a full reaper + worker sweep over pending queue records."""
    cfg = settings or load_config()
    db_path = (
        cfg.pipeline.queue_db_path
        if hasattr(cfg.pipeline, "queue_db_path")
        else "./data/queue.db"
    )
    queue = EpisodeQueue(str(db_path))
    worker = EpisodeWorker(queue, cfg)
    dispatcher = ReportDispatcher.from_settings(cfg)

    start_time = time.monotonic()
    stats_initial = queue.get_stats()

    # Phase 1: Reaper-first (unlock stuck processing rows)
    reaped_count = await queue.reap_stuck_processing(timeout_sec=PROCESSING_TIMEOUT_SECONDS)
    if reaped_count > 0:
        logger.info("Reaped %d stuck processing records.", reaped_count)

    # Phase 2: Claim batch of queued records
    records = await queue.claim_next_batch(limit=batch_size)
    processed_count = len(records)
    succeeded_count = 0
    failed_count = 0

    if records:
        logger.info("Claimed %d records for processing (concurrency=%d)", processed_count, concurrency)
        sem = asyncio.Semaphore(concurrency)

        async def _safe_process(rec: EpisodeRecord):
            nonlocal succeeded_count, failed_count
            # Check soft sweep deadline
            if time.monotonic() - start_time > sweep_timeout_sec:
                logger.warning("Sweep soft timeout reached; deferring record %s", rec.id)
                await queue.mark_failed(rec.id, "Sweep soft timeout exceeded")
                failed_count += 1
                return

            async with sem:
                ok = await worker.process_record(rec, timeout_sec=record_timeout_sec)
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
