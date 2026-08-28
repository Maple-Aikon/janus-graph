"""Periodic queue batch sweeper and DLQ reaper."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional
from ..config import JanusSettings, load_config
from .queue import EpisodeQueue
from .worker import EpisodeWorker

logger = logging.getLogger(__name__)


async def run_cron_sweep(settings: Optional[JanusSettings] = None, batch_size: int = 50) -> dict[str, int]:
    """Drain queued episodes and return summary."""
    cfg = settings or load_config()
    queue = EpisodeQueue(cfg.pipeline.queue_db_path)
    worker = EpisodeWorker(queue, cfg)

    stats_before = queue.get_stats()
    logger.info("Starting cron sweep. Current stats: %s", stats_before)

    # In Phase 1 scaffolding, we verify connectivity & queue query
    return {
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "queued_remaining": stats_before.get("queued", 0),
    }
