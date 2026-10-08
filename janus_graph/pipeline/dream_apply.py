"""FalkorDB execution for the Dream Mode phases that need the graph.

Phase 3 (orphan pruning) lives here; Phase 1 has no execution code because the
clustering algorithm is not reproducible (see :mod:`.dream_phases`).

Safety rules mirror :mod:`.dedup_apply`:
  * Nothing is written unless ``apply=True``. The nightly cron run passes
    force=False and reports what *could* be pruned.
  * Nodes are addressed by ``uuid``, never by name, so a plan computed from one
    snapshot cannot silently re-target a different node if the graph shifts.
  * Deletion re-checks ``NOT (n)--()`` in the SAME statement that deletes, so a
    node which gained an edge between planning and applying is left alone
    rather than detaching that edge. The match is what gates the delete, so
    this needs no transaction (FalkorDB exposes none).
  * Only nodes that are already edge-free are ever passed here, so
    ``DETACH DELETE`` cannot silently eat an edge -- ``DETACH`` is used only to
    survive a concurrent label change, never to discard data on purpose.

Cypher dialect notes for this FalkorDB build:
  * single-quoted strings escape with a BACKSLASH, not SQL doubled quotes, and
    the backslash itself must be escaped FIRST;
  * the driver exposes no ``transaction`` / ``tx`` / ``multi``, so every
    statement is individually atomic and ordering matters.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .dedup_apply import _graph, load_graph_snapshot
from .dream_phases import PrunePlan, plan_orphan_pruning

logger = logging.getLogger("janus_graph.pipeline.dream_apply")

__all__ = ["run_phase1_cluster", "run_phase3_prune"]


def _result_rows(res: Any) -> List[list]:
    rows = getattr(res, "result_set", None)
    if rows is None:
        rows = getattr(res, "set", None) or []
    return [list(r) for r in rows]


def _count_or_zero(res: Any) -> int:
    """Best-effort integer from a ``RETURN count(...) AS n`` result."""
    try:
        rows = _result_rows(res)
        if rows and rows[0]:
            return int(rows[0][0])
    except Exception:  # noqa: BLE001
        return 0
    return 0


#: Two counts, because "0 orphans" is ambiguous on its own: the node may be
#: gone already, or it may have gained an edge since planning. Those are
#: different outcomes and collapsing them silently hides a race.
_EXISTS_COUNT = "MATCH (v:Entity) WHERE v.uuid IN $u RETURN count(v)"

_STILL_ORPHAN_COUNT = (
    "MATCH (v:Entity) WHERE v.uuid IN $u AND NOT (v)--() RETURN count(v)"
)

#: Same gate. Phase D of dedup_apply uses this exact shape, so it is known to
#: parse and run on this FalkorDB; DETACH only ever sees an already-edgeless
#: node, so it cannot silently discard an edge.
_DELETE_ORPHANS = (
    "MATCH (v:Entity) WHERE v.uuid IN $u AND NOT (v)--() DETACH DELETE v"
)


def _delete_orphans(graph: Any, uuids: List[str]) -> Dict[str, Any]:
    """Delete each uuid only if it is still edge-free at delete time.

    The delete statement carries the same ``NOT (v)--()`` gate as the probe, so
    a node that gained an edge in between is skipped rather than detached.
    """
    counters = {
        "deleted": 0,
        "skipped_still_referenced": 0,
        "already_gone": 0,
        "errors": [],
    }
    for uuid in uuids:
        key = {"u": [str(uuid)]}
        try:
            exists = _count_or_zero(graph.query(_EXISTS_COUNT, key))
            if not exists:
                counters["already_gone"] += 1
                continue
            if not _count_or_zero(graph.query(_STILL_ORPHAN_COUNT, key)):
                counters["skipped_still_referenced"] += 1
                continue
            graph.query(_DELETE_ORPHANS, key)
            counters["deleted"] += 1
        except Exception as exc:  # noqa: BLE001
            counters["errors"].append("delete %s: %s" % (str(uuid)[:16], exc))
    return counters

def run_phase1_cluster(
    cfg: Any,
    group_id: str,
    min_degree: Optional[int] = None,
    graph: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute the Phase 1 community partition. Read-only; never mutates.

    Phase 1 has no apply step by design. The partition is a measurement, and
    writing it back would make the graph depend on its own nightly run.

    Args:
        min_degree: override for the low-degree pin threshold. When None the
            live ``cfg`` value is used, so the number in the report is the
            configured one rather than a constant buried in this file.
        graph: an already-open FalkorDB handle; tests inject a fake one.
    """
    from .dedup_apply import _cluster_min_degree
    from .dream_phases import DEFAULT_MIN_DEGREE, build_adjacency, cluster_graph

    threshold = DEFAULT_MIN_DEGREE if min_degree is None else int(min_degree)
    if min_degree is None:
        # Phase 1 reads the same class of setting as Phase 2, so both numbers in
        # a dream report come from one config tree.
        threshold = _cluster_min_degree(cfg)

    owned = False
    if graph is None:
        graph, err = _graph(cfg, group_id)
        if graph is None:
            return {
                "status": "SKIPPED",
                "reason": "FalkorDB unreachable (%s)" % err,
                "detail": "no graph connection",
                "communities": 0,
                "nodes": 0,
            }
        owned = True
    del owned  # read-only: nothing to close that this function opened

    snap, read_err = load_graph_snapshot(graph, group_id)
    if read_err:
        return {
            "status": "FAILED",
            "reason": read_err,
            "detail": "snapshot read failed",
            "communities": 0,
            "nodes": 0,
        }

    result = cluster_graph(
        build_adjacency(snap["entities"], snap["relations"]),
        min_degree=threshold,
    )

    out: Dict[str, Any] = {
        "status": "DONE",
        "detail": result.describe(),
        "applied": False,
        "result": result.as_dict(),
    }
    if not result.converged:
        # Not a failure. Reported so a reader knows the number came from the
        # cycle-resolution rule rather than from a converged fixed point.
        out["reason"] = (
            "label propagation 2-cycled on %d node(s); resolved by the "
            "smallest-uuid rule, so the partition is still reproducible"
            % result.oscillating_nodes
        ) if result.cycle_detected else (
            "did not converge within %d rounds" % result.rounds
        )
    if result.oversized:
        out["oversized"] = True
    return out


def run_phase3_prune(
    cfg: Any,
    group_id: str,
    force: bool = False,
    graph: Optional[Any] = None,
) -> Dict[str, Any]:
    """Plan (and optionally apply) Phase 3 orphan pruning. Never raises.

    Args:
        graph: an already-open FalkorDB handle. Tests inject a fake one; when
            omitted this opens (and owns) its own handle.
    """
    owned = False
    if graph is None:
        graph, err = _graph(cfg, group_id)
        if graph is None:
            return {
                "status": "SKIPPED",
                "reason": "FalkorDB unreachable (%s)" % err,
                "candidates": 0,
                "prunable": 0,
                "deleted": 0,
                "applied": False,
                "detail": "no graph connection",
            }
        owned = True

    try:
        snap, read_err = load_graph_snapshot(graph, group_id)
        if read_err:
            return {
                "status": "FAILED",
                "reason": read_err,
                "candidates": 0,
                "prunable": 0,
                "deleted": 0,
                "applied": False,
                "detail": "snapshot read failed",
            }

        plan: PrunePlan = plan_orphan_pruning(
            snap["entities"], snap["mentions"], snap["relations"]
        )
        prunable = plan.prunable

        out: Dict[str, Any] = {
            "status": "DONE",
            "applied": False,
            "candidates": len(plan.candidates),
            "prunable": len(prunable),
            "protected": plan.protected,
            "protected_chars": plan.protected_chars,
            "deleted": 0,
            "detail": plan.describe(),
            "plan": plan.as_dict(),
        }

        if not prunable:
            # Not a failure: unreferenced-but-valuable is the normal state of a
            # knowledge graph. Saying "0 pruned" without saying why would be
            # indistinguishable from the old bare "DONE".
            out["reason"] = "nothing prunable: all %d orphan(s) hold summary text" % len(
                plan.candidates
            )
            return out

        if not force:
            out["status"] = "PLANNED"
            out["reason"] = "dry-run: %d prunable node(s) withheld (force=False)" % len(
                prunable
            )
            return out

        counters = _delete_orphans(graph, [o.uuid for o in prunable])
        out["applied"] = True
        out["deleted"] = counters["deleted"]
        out["skipped_still_referenced"] = counters["skipped_still_referenced"]
        out["already_gone"] = counters["already_gone"]
        if counters["errors"]:
            out["status"] = "PARTIAL"
            out["reason"] = "; ".join(counters["errors"])[:300]
        elif counters["skipped_still_referenced"]:
            out["status"] = "PARTIAL"
            out["reason"] = "%d node(s) gained an edge and were left alone" % (
                counters["skipped_still_referenced"],
            )
        return out
    finally:
        if owned:
            closer = getattr(graph, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001
                    logger.debug("graph close failed", exc_info=True)
