"""Single episode worker logic with timeout guard and resilient retry/DLQ routing."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Union, cast

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
        # getattr chain: tests build Settings(report=...) with no graphiti
        # section at all, so a direct attribute chain raises AttributeError
        # before add_episode is ever reached.
        _graphiti = getattr(self.settings, "graphiti", None)
        custom_instructions = getattr(_graphiti, "custom_extraction_instructions", None)
        # The `or` short-circuits: settings.graphiti is only read when the
        # payload carries no group_id. Measured 2026-10-10 -- BOTH payload
        # writers (queue.EpisodeQueue.enqueue and
        # daemon.episode_queue_adapter) always write a group_id, so on any
        # real row this fallback is dead code. `cast` is an identity function
        # at runtime, so the AttributeError a contracts.Settings would raise
        # here is preserved exactly; the cast only stops mypy from reporting
        # it. Tests pin that contract
        # (tests/test_extraction_instructions.py::test_worker_survives_
        # settings_without_graphiti_section).
        group_id = payload.get("group_id") or cast(JanusSettings, self.settings).graphiti.group_id
        name = payload.get("name") or f"ep_{record.id[:8]}"
        source_desc = payload.get("source_description", "agent_interaction")

        if not content.strip():
            logger.warning("Episode %s has empty content, skipping.", record.id)
            await self.queue.mark_done(record.id)
            return True

        await self.queue.update_checkpoint(record.id, "ingesting")
        # create_graphiti_instance is typed (settings: Optional[JanusSettings]).
        # Passing a contracts.Settings crashes inside it at
        # `cfg.graphiti.llm.api_key` -- measured 2026-10-10 -- but no
        # production caller does: cli/main.py and daemon/cron_loop.py both pass
        # a JanusSettings, and every test that reaches this line patches
        # create_graphiti_instance. `cast` is an identity function, so the
        # crash and the `should_retry` classification of it (AttributeError ->
        # retry=True, KILL-4) are unchanged.
        client = create_graphiti_instance(cast(JanusSettings, self.settings))

        ref_time = datetime.now(timezone.utc)
        if record.created_at:
            try:
                ref_time = datetime.fromisoformat(record.created_at)
            except Exception as err:
                # Never silently swallow this. The fallback is NOT equivalent:
                # it stamps the episode with ingest time instead of creation
                # time, which quietly corrupts temporal attribution in the
                # graph. A warning is the correct altitude -- raising here
                # would drop an otherwise-ingestable episode on the floor.
                #
                # Deliberately still `Exception`, not `ValueError`: narrowing it
                # would turn a non-str created_at (e.g. an int from a migrated
                # row) from a silent fallback into a hard failure. This change
                # adds observability ONLY; the accepted-value set is unchanged.
                logger.warning(
                    "Episode %s has unparseable created_at %r (%s: %s); "
                    "falling back to ingest time -- temporal attribution may be off",
                    record.id,
                    record.created_at,
                    type(err).__name__,
                    err,
                )

        # L1: only pass the kwarg when configured. Passing it always,
        # even as None, changes the add_episode call shape and breaks
        # tests that pin the exact kwargs.
        ingest_kwargs: dict = {}
        if custom_instructions:
            ingest_kwargs["custom_extraction_instructions"] = custom_instructions

        await client.add_episode(
            name=name,
            episode_body=content,
            source_description=source_desc,
            reference_time=ref_time,
            group_id=group_id,
            **ingest_kwargs,
        )
        await self.queue.mark_done(record.id)
        return True
