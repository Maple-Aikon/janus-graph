"""Single episode worker logic with timeout guard and resilient retry/DLQ routing."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Any, Optional, Union
from ..config import JanusSettings, load_config
from ..core.contracts import Settings
from .queue import EpisodeQueue, EpisodeRecord
from .retry import send_to_dlq, should_retry

logger = logging.getLogger("janus_graph.pipeline.worker")


class EpisodeWorker:
    """Processes a single queued episode with graphiti ingestion and timeout protection."""

    def __init__(
        self,
        queue: EpisodeQueue,
        settings: Optional[Union[JanusSettings, Settings]] = None,
    ):
        self.queue = queue
        self.settings = settings or load_config()

    async def process_record(self, record: EpisodeRecord, timeout_sec: float = 300.0) -> bool:
        """Process a single episode record protected by a timeout."""
        try:
            return await asyncio.wait_for(self._execute_ingestion(record), timeout=timeout_sec)
        except asyncio.TimeoutError as err:
            logger.error("Episode %s timed out after %.1fs", record.id, timeout_sec)
            if should_retry(record, err):
                await self.queue.mark_failed(record.id, f"TimeoutError: exceeded {timeout_sec}s")
            else:
                await send_to_dlq(self.queue, record, err)
            return False
        except Exception as err:
            logger.error("Failed to process episode %s: %s", record.id, err)
            if should_retry(record, err):
                await self.queue.mark_failed(record.id, f"{type(err).__name__}: {err}")
            else:
                await send_to_dlq(self.queue, record, err)
            return False

    async def _execute_ingestion(self, record: EpisodeRecord) -> bool:
        # Lazy import Graphiti instance creation
        from ..core.instance import create_graphiti_instance

        payload = record.payload
        content = payload.get("content", "")
        group_id = payload.get("group_id") or self.settings.graphiti.group_id
        name = payload.get("name") or f"ep_{record.id[:8]}"
        source_desc = payload.get("source_description", "agent_interaction")

        if not content.strip():
            logger.warning("Episode %s has empty content, skipping.", record.id)
            await self.queue.mark_done(record.id)
            return True

        await self.queue.update_checkpoint(record.id, "ingesting")
        client = create_graphiti_instance(self.settings)

        ref_time = datetime.now(timezone.utc)
        if record.created_at:
            try:
                ref_time = datetime.fromisoformat(record.created_at)
            except Exception:
                pass

        await client.add_episode(
            name=name,
            episode_body=content,
            source_description=source_desc,
            reference_time=ref_time,
            group_id=group_id,
        )
        await self.queue.mark_done(record.id)
        return True
