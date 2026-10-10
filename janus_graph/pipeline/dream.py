"""Dream Mode: Nightly Memory Consolidation & Pruning for Janus-Graph (V2.1 Hardened)."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from ..config import JanusSettings, load_config, resolve_home_relative
from ..core.contracts import Settings
from ..report.dispatcher import ReportDispatcher
from ..report.models import ReportSeverity
from .dream_phases import cluster_report
from .queue import EpisodeQueue

logger = logging.getLogger("janus_graph.pipeline.dream")

DEFAULT_COMMUNITY_THRESHOLD = int(os.environ.get("GRAPHITI_DREAM_COMMUNITY_THRESHOLD", "50"))
DEFAULT_TIMEOUT_TOTAL = 210.0  # 3.5 minutes total budget
MAX_LABEL_PROPAGATION_ITERATIONS = 20


def bounded_label_propagation(
    projection: dict[str, list[Any]],
    max_iter: int = MAX_LABEL_PROPAGATION_ITERATIONS,
) -> list[list[str]]:
    """Bounded Label Propagation for Community Detection to prevent infinite oscillations."""
    community_map = {uuid_val: i for i, uuid_val in enumerate(projection.keys())}

    for _ in range(max_iter):
        no_change = True
        new_community_map: dict[str, int] = {}

        for uuid_val, neighbors in projection.items():
            curr_community = community_map[uuid_val]
            community_candidates: dict[int, int] = defaultdict(int)
            for neighbor in neighbors:
                neighbor_uuid = getattr(neighbor, "node_uuid", None) or getattr(
                    neighbor, "uuid", str(neighbor)
                )
                edge_count = getattr(neighbor, "edge_count", 1)
                if neighbor_uuid in community_map:
                    community_candidates[community_map[neighbor_uuid]] += edge_count

            community_lst = [(count, comm) for comm, count in community_candidates.items()]
            community_lst.sort(reverse=True)
            candidate_rank, community_candidate = community_lst[0] if community_lst else (0, -1)
            if community_candidate != -1 and candidate_rank > 1:
                new_community = community_candidate
            else:
                new_community = max(community_candidate, curr_community)

            new_community_map[uuid_val] = new_community
            if new_community != curr_community:
                no_change = False

        community_map = new_community_map
        if no_change:
            break

    community_cluster_map: dict[int, list[str]] = defaultdict(list)
    for uuid_val, community in community_map.items():
        community_cluster_map[community].append(uuid_val)

    return list(community_cluster_map.values())


def _phase1_result(phase1: Dict[str, Any]) -> Any:
    """Rebuild a ClusterResult from a phase-1 payload, or an empty one.

    ``run_phase1_cluster`` returns the partition as plain dicts so the payload
    survives JSON serialisation into the dream report. ``cluster_report`` needs
    the dataclass for ``describe()``.
    """
    from .dream_phases import ClusterResult

    data = phase1.get("result")
    if not isinstance(data, dict):
        return ClusterResult(
            nodes=0,
            communities=0,
            rounds=0,
            converged=True,
            cycle_detected=False,
            oscillating_nodes=0,
            pinned_nodes=0,
            components=0,
            largest_component=0,
            largest_community=0,
            min_degree=0,
            partition_hash="",
            oversized=False,
        )
    fields = {f for f in ClusterResult.__dataclass_fields__}
    return ClusterResult(**{k: v for k, v in data.items() if k in fields})


async def run_dream_consolidation(
    settings: Optional[Union[JanusSettings, Settings]] = None,
    force: bool = False,
    group_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute Dream Mode memory consolidation phases with full undo log."""
    cfg = settings or load_config()
    # contracts.Settings keeps the queue path under ``paths``, not
    # ``pipeline``, so the old hasattr guard was False for that type and
    # the relative literal WAS reachable here: dream runs to completion on a
    # bare contracts.Settings and opened <cwd>/data/queue.db. Anchor at home.
    pipeline_path = getattr(cfg.pipeline, "queue_db_path", None)
    if pipeline_path is None:
        # getattr() guards only the attribute *inside* it — `cfg.paths` is
        # evaluated first, and JanusSettings has no `paths`, so the unguarded
        # form raised AttributeError exactly when it was needed. Same defect
        # class as cron.py:74; fixed there 2026-10-10, missed here.
        paths_cfg = getattr(cfg, "paths", None)
        pipeline_path = getattr(paths_cfg, "queue_db_path", None)
    if pipeline_path is None:
        pipeline_path = "./data/queue.db"
    db_path = resolve_home_relative(str(pipeline_path))
    queue = EpisodeQueue(db_path)
    dispatcher = ReportDispatcher.from_settings(cfg)

    run_id = str(uuid.uuid4())
    start_time = time.monotonic()
    now = datetime.now(timezone.utc).isoformat()

    results: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": now,
        "status": "running",
        "phase_1_clustering": "SKIPPED",
        "phase_2_deduplication": "PENDING",
        "phase_3_orphan_pruning": "PENDING",
        "phase_4_dlq_repair": "PENDING",
        "nodes_before": 0,
        "nodes_after": 0,
        "duration_ms": 0,
    }

    # The memory tenant is shared by every phase that reads the graph. Hoisted
    # ABOVE phase 1: the previous ordering computed it inside phase 2, so a
    # phase 1 reference to it is a NameError at runtime, not a style nit.
    dedup_group = (
        group_id or getattr(getattr(cfg, "graphiti", None), "group_id", None) or "graphiti_memory"
    )

    # Phase 2's own write switch. Read from ``cfg.pipeline.dream.dedup_apply``
    # with the same defensive getattr chain the threshold and min-degree
    # resolvers use, so a bare ``contracts.Settings`` (no ``pipeline.dream``)
    # still yields False instead of raising. Absent means False — dry-run —
    # which is the behaviour that shipped until now.
    dedup_apply = bool(
        getattr(
            getattr(getattr(cfg, "pipeline", None), "dream", None),
            "dedup_apply",
            False,
        )
    )
    results["phase_2_apply"] = dedup_apply

    # Phase 1: Community clustering. Runs a DETERMINISTIC partition: synchronous
    # label propagation seeded from a lexicographic total order, with 2-cycles
    # resolved by a fixed smallest-uuid rule. The old `bounded_label_propagation`
    # was reported as "DONE" while only counting episodes, and the version that
    # followed refused to run at all; both are replaced by a real measurement
    # plus the structural floor it must be read against (see dream_phases for
    # the two falsified hypotheses behind the old diagnosis).
    cluster_detail = "not attempted: episode count unavailable"
    try:
        from .dream_apply import run_phase1_cluster

        stats = queue.get_stats()
        total_episodes = stats.get("done", 0) + stats.get("queued", 0)
        phase1 = run_phase1_cluster(cfg, dedup_group)
        cluster_detail = cluster_report(
            _phase1_result(phase1), total_episodes, DEFAULT_COMMUNITY_THRESHOLD, force
        )
        results["phase_1_clustering"] = cluster_detail
        results["phase_1_detail"] = phase1
        results["phase_1_communities"] = phase1.get("communities", 0)
        logger.info(
            "dream: phase_1_clustering %s communities=%s",
            phase1.get("status", "FAILED"),
            phase1.get("communities", 0),
        )
    except Exception as err:  # noqa: BLE001 - clustering failure must not kill the run
        logger.warning("Clustering phase error: %s", err)
        cluster_detail = "FAILED: %s" % err
        results["phase_1_clustering"] = cluster_detail

    # Phase 2: Entity Deduplication. Reads the graph, plans merges, and writes
    # ONLY when ``force`` is set — the nightly cron run passes force=False, so
    # it reports what it would do without mutating anything. Previously this
    # line was the literal ``results["phase_2_deduplication"] = "DONE"`` with no
    # implementation behind it, which made a no-op indistinguishable from real
    # work in every dream report since 2026-08-28. ``dedup_group`` is now
    # resolved once above Phase 1 and shared.
    #
    # ``force`` is NOT reused here. It reaches phase 3 as well (below), so
    # using it to make dedup write would also enable orphan pruning — 38 nodes
    # on the 2026-10-07 snapshot. ``dedup_apply`` is phase 2's own switch.
    try:
        from .dedup_apply import run_phase2_dedup

        phase2 = run_phase2_dedup(cfg, dedup_group, force=force or dedup_apply)
        phase2_status = phase2.get("status", "FAILED")
        # Show the reason too. `detail or reason` dropped ``reason`` whenever
        # both were present, which is exactly the case that matters: a SKIPPED
        # phase with a diagnostic reason then reported only its generic detail.
        detail = " - ".join(x for x in (phase2.get("detail"), phase2.get("reason")) if x)
        results["phase_2_deduplication"] = "%s (%s)" % (phase2_status, detail)
        results["phase_2_detail"] = phase2
        results["phase_2_merges"] = phase2.get("merges", 0)
        # Real graph size, not the hardcoded 0 this used to report. Taken from
        # the snapshot Phase 2 already read, so it costs no extra query.
        seen = phase2.get("entities_seen")
        if isinstance(seen, int):
            results["nodes_before"] = seen
        logger.info(
            "dream: phase_2_deduplication %s merges=%s applied=%s",
            phase2_status,
            phase2.get("merges", 0),
            phase2.get("applied", False),
        )
    except Exception as err:  # noqa: BLE001 - a dedup failure must not kill the run
        logger.warning("Deduplication phase error: %s", err)
        results["phase_2_deduplication"] = "FAILED: %s" % err

    # Phase 3: True Orphan Node Pruning. Unreferenced is NOT the same as
    # disposable: on the 2026-10-07 snapshot all 10 orphans hold 84-251 chars of
    # summary each, so a name-only "orphan" rule would have deleted knowledge.
    # The plan deletes only nodes that are unreferenced AND empty, and reports
    # the rest. Writes only under ``force``. This used to be the literal "DONE".
    try:
        from .dream_apply import run_phase3_prune

        phase3 = run_phase3_prune(cfg, dedup_group, force=force)
        phase3_status = phase3.get("status", "FAILED")
        detail3 = " - ".join(x for x in (phase3.get("detail"), phase3.get("reason")) if x)
        results["phase_3_orphan_pruning"] = "%s (%s)" % (phase3_status, detail3)
        results["phase_3_detail"] = phase3
        results["phase_3_pruned"] = phase3.get("deleted", 0)
        # Only moves when a prune actually happened and reported a count.
        pruned = phase3.get("deleted")
        if isinstance(pruned, int):
            results["nodes_after"] = max(0, results.get("nodes_before", 0) - pruned)
        logger.info(
            "dream: phase_3_orphan_pruning %s candidates=%s deleted=%s applied=%s",
            phase3_status,
            phase3.get("candidates", 0),
            phase3.get("deleted", 0),
            phase3.get("applied", False),
        )
    except Exception as err:  # noqa: BLE001 - a prune failure must not kill the run
        logger.warning("Orphan pruning phase error: %s", err)
        results["phase_3_orphan_pruning"] = "FAILED: %s" % err

    # Phase 4: DLQ Auto-repair — requeue failed/aborted (cheap, on episodes table)
    #         + DLQ batch replay (filter by class, on dead_letter table).
    try:
        repaired_count = await queue.reap_failed_or_aborted(limit=100)
        results["phase_4_dlq_repair"] = f"DONE ({repaired_count} requeued)"

        # DLQ batch replay — controlled by cfg.pipeline.dlq_replay
        dlq_cfg = getattr(cfg.pipeline, "dlq_replay", None)
        if dlq_cfg is not None and getattr(dlq_cfg, "enabled", True):
            replayed = await queue.replay_dlq_batch(
                limit=getattr(dlq_cfg, "limit", 100),
                classes=getattr(dlq_cfg, "classes", None),
                min_age_sec=getattr(dlq_cfg, "min_age_sec", 60),
            )
            results["phase_4_replay_dlq_batch"] = (
                f"DONE ({replayed} replayed, classes={getattr(dlq_cfg, 'classes', [])})"
            )
            if replayed > 0:
                logger.info("dream: phase_4_replay_dlq_batch replayed=%d", replayed)
        else:
            results["phase_4_replay_dlq_batch"] = "SKIPPED (dlq_replay.enabled=false)"
    except Exception as err:
        results["phase_4_dlq_repair"] = f"FAILED: {err}"

    # "status" used to be the unconditional literal "completed", so a run in
    # which every phase threw still reported success. Now it is derived: any
    # phase that actually failed downgrades the whole run.
    failed_phases = [
        k for k in results if k.startswith("phase_") and str(results[k]).startswith("FAILED")
    ]
    if failed_phases:
        results["status"] = "completed_with_failures"
        results["failed_phases"] = failed_phases
    else:
        results["status"] = "completed"
    results["duration_ms"] = round((time.monotonic() - start_time) * 1000, 2)
    results["completed_at"] = datetime.now(timezone.utc).isoformat()

    # Dispatch dream consolidation report
    try:
        await asyncio.shield(
            dispatcher.emit_quick(
                kind="dream_consolidation",
                summary=f"Dream Mode completed in {results['duration_ms']}ms (run_id={run_id[:8]})",
                severity=ReportSeverity.INFO,
                details=results,
            )
        )
    except Exception as err:
        logger.warning("Failed to emit dream report: %s", err)

    return results
