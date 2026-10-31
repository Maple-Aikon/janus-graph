"""Single episode worker logic with timeout guard and resilient retry/DLQ routing."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union, cast

from ..config import JanusSettings, load_config
from ..core.contracts import Settings
from ..core.payload_hash import canonical_payload_hash
from .queue import EpisodeQueue, EpisodeRecord
from .retry import describe_exception, send_to_dlq, should_retry

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
        # The backstop runs FIRST, before any Graphiti call. It is
        # opt-in and default-off; when it skips a row it returns True
        # (the record is handled — no ingestion needed) and the row has
        # already been marked with its own terminal status.
        if await self._backstop_skip_if_ingested(record):
            return True
        try:
            return await asyncio.wait_for(self._execute_ingestion(record), timeout=timeout_sec)
        except asyncio.TimeoutError as err:
            logger.error("Episode %s timed out after %.1fs", record.id, timeout_sec)
            if should_retry(record, err):
                await self.queue.mark_failed(record.id, describe_exception(err))
            else:
                await send_to_dlq(self.queue, record, err)
            return False
        except Exception as err:
            logger.error("Failed to process episode %s: %s", record.id, err)
            if should_retry(record, err):
                await self.queue.mark_failed(record.id, describe_exception(err))
            else:
                await send_to_dlq(self.queue, record, err)
            return False

    async def _backstop_skip_if_ingested(self, record: EpisodeRecord) -> bool:
        """Skip this row when its payload was ALREADY ingested.

        THE BACKSTOP, NOT THE GATE
        --------------------------
        The enqueue gate is the primary dedup mechanism: it is the only
        place where the writer can still tell the caller "this is a
        duplicate, and here is the id of the row that holds it". This
        method is a SECOND line of defence that catches what the gate
        structurally cannot:

          * the CROSS-PROCESS race — two writers both passed the gate's
            check before either inserted (the gate's check-then-insert is
            a TOCTOU window across three writer processes);
          * the "same content sent again hours later" case — by then the
            first row has left the active set, so the gate's active-only
            lookup finds nothing.

        IT IS OPT-IN AND DEFAULT-OFF. Done-aware dedup can suppress a
        LEGITIMATE re-ingestion (the same text re-sent because the user
        wants it in the graph twice), and that is a data-loss failure. So
        it must be a deliberate choice, not a default.

        IT NEVER RAISES. This is load-bearing: the worker's failure path
        routes to the DLQ, so an exception here would turn a benign skip
        into queue pressure and DLQ noise. Every failure inside this
        method degrades to "do not skip" — the episode is processed
        normally, exactly as if the backstop were off.

        The row is marked with a dedicated terminal status
        (``skipped_duplicate``), never ``done`` and never a failure
        status — see ``EpisodeQueue.mark_skipped_duplicate``.
        """
        try:
            backstop = self._resolve_backstop()
            if not backstop.get("enabled"):
                return False

            payload = record.payload
            content = payload.get("content")
            group_id = payload.get("group_id")
            source_description = payload.get("source_description")
            if content is None or group_id is None or source_description is None:
                # Cannot compute the hash — do not skip.
                return False

            payload_hash = canonical_payload_hash(
                content, group_id, source_description
            )
            existing = await asyncio.to_thread(
                self.queue.find_ingested_duplicate_sync, payload_hash
            )
            if existing is None:
                return False
            if existing == record.id:
                # The row IS the ingested one — never skip a row against
                # itself.
                return False

            logger.info(
                "backstop: skipping episode %s — payload already ingested as %s",
                record.id,
                existing,
            )
            await self.queue.mark_skipped_duplicate(
                record.id,
                existing_episode_id=existing,
                payload_hash=payload_hash,
            )
            return True
        except Exception as err:  # pragma: no cover - defensive only
            # Any failure at all degrades to "do not skip".
            logger.warning(
                "backstop check failed for episode %s (%s) — processing normally",
                record.id,
                err,
            )
            return False

    def _resolve_backstop(self) -> Dict[str, Any]:
        """Read the backstop settings, tolerating any settings shape.

        ``JanusSettings`` carries ``pipeline.dedup.backstop``;
        ``contracts.Settings`` does not have that subtree at all (tests
        build one). Reading it through ``getattr`` chains means an
        absent section degrades to "disabled" instead of raising.
        """
        pipeline_cfg = getattr(self.settings, "pipeline", None)
        dedup_cfg = getattr(pipeline_cfg, "dedup", None)
        backstop = getattr(dedup_cfg, "backstop", None)
        if backstop is None:
            return {"enabled": False}
        if isinstance(backstop, dict):
            return backstop
        enabled = getattr(backstop, "enabled", False)
        return {"enabled": bool(enabled)}

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
