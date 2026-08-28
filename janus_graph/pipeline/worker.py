"""Single episode worker logic."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional
from ..config import JanusSettings, load_config
from .queue import EpisodeQueue, EpisodeRecord

logger = logging.getLogger(__name__)


class EpisodeWorker:
    """Processes a single queued episode with graphiti ingestion."""

    def __init__(self, queue: EpisodeQueue, settings: Optional[JanusSettings] = None):
        self.queue = queue
        self.settings = settings or load_config()

    async def process_record(self, record: EpisodeRecord) -> bool:
        """Process a single episode record."""
        # Lazy import to avoid loading graphiti on light startup
        from ..core.instance import create_graphiti_instance

        try:
            payload = json.loads(record.payload_json)
            content = payload.get("content", "")
            group_id = payload.get("group_id") or self.settings.graphiti.group_id

            client = create_graphiti_instance(self.settings)
            await client.add_episode(
                name=f"ep_{record.id[:8]}",
                episode_body=content,
                source_description="agent_interaction",
                group_id=group_id,
            )
            return True
        except Exception as err:
            logger.error("Failed to process episode %s: %s", record.id, err)
            return False
