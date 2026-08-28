"""Retry policies and Dead-Letter Queue (DLQ) helpers."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from .queue import EpisodeQueue, EpisodeRecord

logger = logging.getLogger("janus_graph.pipeline.retry")

NON_RETRYABLE_SCHEMA_SIGNATURES = (
    "ExtractedEdges()",
    "ExtractedEntities()",
    "entity_resolutions",
    "duplicate_facts",
    "SummarizedEntities",
    "argument after ** must be a mapping",
)


def compute_backoff_seconds(attempt: int, base: float = 2.0, factor: float = 2.0) -> float:
    """Compute exponential backoff in seconds."""
    delay = base * (factor ** max(0, attempt - 1))
    return min(delay, 300.0)


def should_retry(record: EpisodeRecord, exc: BaseException, max_attempts: int = 3) -> bool:
    """Determine whether an episode should be retried or sent to DLQ."""
    if record.attempt_count + 1 >= max_attempts:
        return False

    # Always-fatal standard exceptions
    if isinstance(exc, (TypeError, ValueError, KeyError)):
        return False

    name = type(exc).__name__
    msg = str(exc)

    # Unrecoverable validation errors
    if name == "ValidationError" or any(sig in msg for sig in NON_RETRYABLE_SCHEMA_SIGNATURES):
        logger.info(
            "Non-retryable schema error detected: %s (signatures: %s)",
            name,
            [s for s in NON_RETRYABLE_SCHEMA_SIGNATURES if s in msg],
        )
        return False

    return True


async def send_to_dlq(queue: EpisodeQueue, record: EpisodeRecord, exc: BaseException) -> None:
    """Mark record aborted and write into DLQ."""
    error_msg = f"{type(exc).__name__}: {exc}"
    await queue.mark_aborted(record.id, error_msg)
    logger.warning("Episode %s routed to DLQ: %s", record.id, error_msg)
