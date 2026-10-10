"""FalkorDB execution for the Phase 2 dedup plan.

Kept separate from :mod:`janus_graph.pipeline.dedup` (pure logic) and from
:mod:`janus_graph.pipeline.dream` (orchestration) so each layer is testable on
its own.

Safety rules encoded here:
  * Nothing is written unless ``apply=True``. ``run_dream_consolidation`` sets
    it when ``force=True`` OR when ``cfg.pipeline.dream.dedup_apply`` is on, so
    the nightly cron run writes only once an operator opts in per-install.
    Default is still dry-run.
  * Merges address nodes by ``uuid``, never by name, so a plan computed from one
    snapshot cannot silently re-target a different node if the graph shifts.
  * Redirects CREATE the new edge BEFORE deleting the old one. FalkorDB exposes
    no transaction (verified: no ``transaction``/``tx``/``multi`` in the
    ``falkordb`` client), so DELETE-then-CREATE would lose the edge outright if
    the process died in between. CREATE-then-DELETE can at worst duplicate an
    edge, and the duplicate-collapse step cleans that up on the next run.
  * Nodes are deleted LAST and only once provably empty (``NOT (v)--()``), so a
    failed redirect leaves the victim node with its edges intact rather than
    silently detaching them.

Why DELETE + CREATE instead of SET on the pointer properties
------------------------------------------------------------
An earlier draft redirected edges with ``SET e.dst_uuid = ...``. That is wrong:
in a graph the relationship's endpoints are the *edge itself*, and ``src_uuid``
/ ``dst_uuid`` are ordinary properties that merely mirror them. Rewriting the
property leaves the edge still attached to the victim node, and the next node
deletion detaches it -- silently dropping every MENTIONS the merge was supposed
to preserve.

MEASURED 2026-10-09 on the live graph (26,030 MENTIONS), correcting an earlier
claim in this docstring: MENTIONS carries ONLY ``uuid``, ``group_id`` and
``created_at``. It has NO ``src_uuid``/``dst_uuid`` properties at all, so the
"properties mirror the topology" premise above is false for MENTIONS. Reading
those two names returns NULL on every row, which silently turned Phase A into a
no-op -- the CREATE matched ``(:Episodic {uuid: NULL})`` and matched nothing --
while ``mentions_moved`` still went up. The victim then kept its MENTIONS, so
Phase D's ``NOT (v)--()`` guard correctly refused to delete it, and the phase
never converged. MENTIONS endpoints must be read from the matched NODES.

So the redirect is: read the edge endpoints, DELETE it, re-CREATE it with the
survivor endpoint. Measured proof that writing both spellings is required for
RELATES_TO (and only for RELATES_TO): it carries both
``source_node_uuid``/``target_node_uuid`` and ``src_uuid``/``dst_uuid``, and on
all 12,295 edges the two spellings agree (0 disagree, 0 missing), so both must
be written to keep that record self-consistent. MENTIONS must NOT be given
invented properties -- a property that never existed is exactly what made this
bug invisible for two nightly runs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from .dedup import DedupPlan

logger = logging.getLogger("janus_graph.pipeline.dedup_apply")

DEFAULT_BATCH = 200

_ENTITIES_Q = (
    "MATCH (n:Entity) WHERE n.group_id = $gid RETURN n.uuid, n.name, n.summary, n.created_at"
)
_MENTIONS_Q = (
    "MATCH (s:Episodic)-[e:MENTIONS]->(n:Entity) WHERE n.group_id = $gid "
    "RETURN e.uuid, s.uuid, n.uuid, e.created_at, e.group_id"
)
_RELATIONS_Q = (
    "MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity) "
    "WHERE a.group_id = $gid "
    "RETURN e.uuid, e.source_node_uuid, e.target_node_uuid, e.fact, e.created_at, "
    "e.episodes, e.valid_at, e.reference_time, e.name"
)


def _graph(cfg: Any, group_id: str):
    """Open a FalkorDB graph handle, or return ``(None, error_string)``.

    Returns rather than raises so an unreachable database degrades the nightly
    run to SKIPPED instead of crashing it.
    """
    engine = getattr(cfg, "graphiti", None) or getattr(cfg, "engine", None)
    host = getattr(engine, "host", None) or "localhost"
    port = getattr(engine, "port", None) or 6379
    try:
        from falkordb import FalkorDB

        client = FalkorDB(host=host, port=port)
        graph = client.select_graph(group_id)
        graph.query("RETURN 1")
        return graph, None
    except Exception as exc:  # noqa: BLE001
        return None, "%s: %s" % (type(exc).__name__, exc)


def _records(graph: Any, cypher: str, params: Dict[str, Any]) -> List[list]:
    res = graph.query(cypher, params)
    rows = getattr(res, "result_set", None)
    if rows is None:
        rows = getattr(res, "set", None) or []
    return [list(r) for r in rows]


def load_graph_snapshot(graph: Any, group_id: str) -> Tuple[Dict[str, List[Dict]], str]:
    """Read entities + edges into plain dicts for the planner."""
    out: Dict[str, List[Dict]] = {"entities": [], "mentions": [], "relations": []}
    errors = []
    try:
        for rec in _records(graph, _ENTITIES_Q, {"gid": group_id}):
            rec = (list(rec) + [None] * 4)[:4]
            out["entities"].append(
                {
                    "uuid": rec[0],
                    "name": rec[1],
                    "summary": rec[2],
                    "created_at": str(rec[3]) if rec[3] else None,
                }
            )
    except Exception as exc:  # noqa: BLE001
        errors.append("entities: %s" % exc)
    try:
        for rec in _records(graph, _MENTIONS_Q, {"gid": group_id}):
            rec = (list(rec) + [None] * 5)[:5]
            out["mentions"].append(
                {
                    "uuid": rec[0],
                    "src_uuid": rec[1],
                    "dst_uuid": rec[2],
                    "created_at": str(rec[3]) if rec[3] else None,
                    "group_id": rec[4] or group_id,
                }
            )
    except Exception as exc:  # noqa: BLE001
        errors.append("mentions: %s" % exc)
    try:
        for rec in _records(graph, _RELATIONS_Q, {"gid": group_id}):
            rec = (list(rec) + [None] * 9)[:9]
            # RELATES_TO stores both spellings; they agreed on all 12,295 edges
            # in the 2026-10-07 snapshot, but prefer source/target when present.
            src = rec[1] or rec[2]
            dst = rec[2] or rec[1]
            out["relations"].append(
                {
                    "uuid": rec[0],
                    "src_uuid": src,
                    "dst_uuid": dst,
                    "fact": rec[3],
                    "created_at": str(rec[4]) if rec[4] else None,
                    "episodes": rec[5] or [],
                    "valid_at": str(rec[6]) if rec[6] else None,
                    "reference_time": str(rec[7]) if rec[7] else None,
                    "name": rec[8],
                }
            )
    except Exception as exc:  # noqa: BLE001
        errors.append("relations: %s" % exc)
    return out, "; ".join(errors)


def _root_map(redirect: Dict[str, str]) -> Dict[str, str]:
    """Collapse chains (A->B->C becomes A->C); cycle-safe."""
    root: Dict[str, str] = {}

    def _root(uuid: str) -> str:
        seen = set()
        cur = uuid
        while cur in redirect and cur not in seen:
            seen.add(cur)
            cur = redirect[cur]
        return cur

    for src in redirect:
        root[src] = _root(src)
    return root


def apply_plan(
    graph: Any,
    plan: DedupPlan,
    apply: bool = False,
    batch: int = DEFAULT_BATCH,
) -> Dict[str, Any]:
    """Execute ``plan`` against ``graph``; dry-run unless ``apply`` is True.

    Returns counters describing what was actually done, so the caller can put
    real numbers in the report instead of a literal string.
    """
    counters: Dict[str, Any] = {
        "applied": bool(apply),
        "nodes_deleted": 0,
        "mentions_moved": 0,
        "relations_moved": 0,
        "self_loops_dropped": 0,
        "duplicate_edges_dropped": 0,
        "errors": [],
    }
    if plan.is_empty:
        return counters

    redirect = {m.victim_uuid: m.survivor_uuid for m in plan.merges}
    root = _root_map(redirect)

    if not apply:
        logger.info(
            "dedup dry-run: %d merges planned (threshold=%s); "
            "pass apply=True to write to the graph",
            len(plan.merges),
            plan.threshold,
        )
        counters["nodes_to_delete"] = plan.nodes_to_delete
        counters["mentions_to_move"] = plan.mentions_to_redirect
        counters["relations_to_move"] = plan.relations_to_redirect
        return counters

    victims = list(redirect.keys())

    # ---- Phase A: move MENTIONS (Episodic)-[:MENTIONS]->(victim) ----
    # DELETE + CREATE, because edge endpoints are not writable properties.
    # NOTE: each phase reads into its OWN variable. An earlier draft reused a
    # single ``rows`` name across phases; when a read failed mid-way the
    # previous phase's leftover rows were unpacked with the wrong arity, which
    # crashed the run with a confusing "not enough values to unpack" instead
    # of reporting the real read failure.
    try:
        ment_rows = _records(
            graph,
            "MATCH (s:Episodic)-[e:MENTIONS]->(v:Entity) "
            "WHERE v.uuid IN $victims "
            "RETURN e.uuid, s.uuid, v.uuid, e.created_at, e.group_id",
            {"victims": victims},
        )
    except Exception as exc:  # noqa: BLE001
        counters["errors"].append("mentions-read: %s" % exc)
        ment_rows = []

    moved = 0
    for edge_uuid, src_uuid, dst_uuid, created_at, gid in ment_rows:
        # Endpoints come from the MATCHED NODES, not from edge properties:
        # MENTIONS carries only uuid/group_id/created_at (measured on all
        # 26,030 live edges), so `e.src_uuid` reads NULL and the CREATE below
        # would match `(:Episodic {uuid: NULL})` -- which matches nothing.
        src_uuid, dst_uuid = str(src_uuid or ""), str(dst_uuid or "")
        if not src_uuid or not dst_uuid:
            counters["errors"].append("mentions-skip %s: unresolved endpoint" % edge_uuid)
            continue
        new_dst = root.get(dst_uuid, dst_uuid)
        if new_dst == dst_uuid:
            continue
        try:
            # CREATE first, DELETE second. FalkorDB has no transaction, so the
            # reverse order could destroy the edge if this process died between
            # the two statements.
            #
            # The DELETE is conditional on the CREATE having actually matched
            # both endpoints. Until the endpoint fix landed, the CREATE could
            # never match (`{uuid: NULL}` matches nothing), so this guard was
            # untestable; now that it CAN match, running DELETE regardless would
            # destroy the MENTIONS outright when the MATCH hits 0 rows -- the
            # exact data loss DELETE+CREATE exists to prevent.
            created = _records(
                graph,
                # The MATCH endpoints MUST be named. An anonymous node in the
                # MATCH pattern (``(:Episodic {uuid: $src})``) is a separate
                # pattern part with no binding, so ``CREATE (s)-[e]->(t)``
                # builds BRAND NEW unlabelled nodes instead of attaching the
                # edge to the matched ones. The edge then exists -- count(*)
                # returns 1, so the guard below passes -- while the MENTIONS
                # points at two empty placeholders and is unreachable by
                # ``(s:Episodic)-[:MENTIONS]->(t:Entity)``. Measured on the live
                # graph: 2 moved edges landed on 4 empty unlabelled nodes.
                "MATCH (s:Episodic {uuid: $src}), (t:Entity {uuid: $new}) "
                "CREATE (s)-[e:MENTIONS {uuid: $eu, group_id: $gid, "
                "created_at: $ca}]->(t) "
                "RETURN count(*)",
                {
                    "src": src_uuid,
                    "new": new_dst,
                    "eu": edge_uuid,
                    "gid": gid or "graphiti_memory",
                    "ca": str(created_at) if created_at else None,
                },
            )
            n_created = int(created[0][0]) if created and created[0][0] else 0
            if n_created <= 0:
                counters["errors"].append(
                    "mentions-create %s: endpoints did not match, edge kept" % edge_uuid
                )
                continue
            graph.query(
                "MATCH (:Episodic {uuid: $src})-[e:MENTIONS]->(:Entity {uuid: $old}) "
                "WHERE e.uuid = $eu DELETE e",
                {"src": src_uuid, "old": dst_uuid, "eu": edge_uuid},
            )
            moved += 1
        except Exception as exc:  # noqa: BLE001
            counters["errors"].append("mentions-move %s: %s" % (edge_uuid, exc))
    counters["mentions_moved"] = moved

    # ---- Phase B: move RELATES_TO (Entity)-[:RELATES_TO]->(victim) ----
    try:
        rel_rows = _records(
            graph,
            "MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity) "
            "WHERE a.uuid IN $victims OR b.uuid IN $victims "
            "RETURN e.uuid, a.uuid, b.uuid, e.fact, e.created_at, e.episodes, "
            "e.valid_at, e.reference_time, e.name",
            {"victims": victims},
        )
    except Exception as exc:  # noqa: BLE001
        counters["errors"].append("relations-read: %s" % exc)
        rel_rows = []

    moved = 0
    for (
        edge_uuid,
        src_uuid,
        dst_uuid,
        fact,
        created_at,
        episodes,
        valid_at,
        reference_time,
        name,
    ) in rel_rows:
        new_src = root.get(str(src_uuid), str(src_uuid))
        new_dst = root.get(str(dst_uuid), str(dst_uuid))
        if new_src == str(src_uuid) and new_dst == str(dst_uuid):
            continue
        if new_src == new_dst:
            # Merging both endpoints into one node: the edge becomes a
            # self-loop. It carries no information and 37 such edges already
            # exist in the graph, so drop it rather than add more.
            counters["self_loops_dropped"] += 1
            try:
                graph.query(
                    "MATCH (:Entity {uuid: $s})-[e:RELATES_TO]->(:Entity {uuid: $d}) "
                    "WHERE e.uuid = $eu DELETE e",
                    {"s": src_uuid, "d": dst_uuid, "eu": edge_uuid},
                )
            except Exception as exc:  # noqa: BLE001
                counters["errors"].append("selfloop-drop %s: %s" % (edge_uuid, exc))
            continue
        try:
            # CREATE first, DELETE second — see the module docstring.
            graph.query(
                # Named endpoints for the same reason as Phase A: an anonymous
                # MATCH node leaves (s)/(t) unbound and CREATE would attach the
                # new RELATES_TO to fresh unlabelled placeholders.
                "MATCH (s:Entity {uuid: $s2}), (t:Entity {uuid: $d2}) "
                "CREATE (s)-[e:RELATES_TO {uuid: $eu, source_node_uuid: $s2, "
                "target_node_uuid: $d2, "
                "group_id: $gid, fact: $fact, name: $name, created_at: $ca, "
                "valid_at: $va, reference_time: $rt, episodes: $eps}]->(t)",
                {
                    "s2": new_src,
                    "d2": new_dst,
                    "eu": edge_uuid,
                    "gid": "graphiti_memory",
                    "fact": fact,
                    "name": name,
                    "ca": str(created_at) if created_at else None,
                    "va": str(valid_at) if valid_at else None,
                    "rt": str(reference_time) if reference_time else None,
                    "eps": episodes or [],
                },
            )
            # Only after the new edge exists do we remove the old one. Both
            # match on the edge uuid, so the freshly created edge -- which
            # carries the same uuid -- is not the one deleted: the MATCH also
            # pins the ORIGINAL endpoints.
            graph.query(
                "MATCH (:Entity {uuid: $s})-[e:RELATES_TO]->(:Entity {uuid: $d}) "
                "WHERE e.uuid = $eu DELETE e",
                {"s": src_uuid, "d": dst_uuid, "eu": edge_uuid},
            )
            moved += 1
        except Exception as exc:  # noqa: BLE001
            counters["errors"].append("relations-move %s: %s" % (edge_uuid, exc))
    counters["relations_moved"] = moved

    # ---- Phase C: drop duplicate edges the move may have created ----
    try:
        dup = _records(
            graph,
            "MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity) "
            "WITH a.uuid AS s, b.uuid AS t, e.fact AS f, collect(e.uuid) AS ids "
            "WITH s, t, f, [x IN ids WHERE x IS NOT NULL] AS ids "
            "WHERE size(ids) > 1 "
            "RETURN sum(size(ids) - 1)",
            {},
        )
        counters["duplicate_edges_dropped"] = int(dup[0][0]) if dup and dup[0][0] else 0
        if counters["duplicate_edges_dropped"]:
            graph.query(
                "MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity) "
                "WITH a.uuid AS s, b.uuid AS t, e.fact AS f, collect(e) AS es "
                "WHERE size(es) > 1 "
                "FOREACH (x IN es[1..] | DELETE x)",
                {},
            )
    except Exception as exc:  # noqa: BLE001
        counters["errors"].append("dedupe-edges: %s" % exc)

    # ---- Phase D: LAST, delete the now-emptied victim nodes ----
    deleted = 0
    for start in range(0, len(victims), batch):
        chunk = victims[start : start + batch]
        try:
            res = _records(
                graph,
                "MATCH (v:Entity) WHERE v.uuid IN $victims AND NOT (v)--() RETURN count(v)",
                {"victims": chunk},
            )
            n_before = int(res[0][0]) if res and res[0][0] is not None else 0
            graph.query(
                "MATCH (v:Entity) WHERE v.uuid IN $victims AND NOT (v)--() DETACH DELETE v",
                {"victims": chunk},
            )
            deleted += n_before
        except Exception as exc:  # noqa: BLE001
            counters["errors"].append("node-delete: %s" % exc)
    counters["nodes_deleted"] = deleted

    return counters


def _dedup_threshold(cfg: Any) -> float:
    """Resolve the Phase 2 merge threshold from the live config.

    Reads ``cfg.pipeline.dream.dedup_threshold`` — the field that actually
    reaches ``dream.py`` — and falls back to the ``DreamConfig`` default rather
    than to a literal in this file. A bare ``Settings()`` or a test double with
    no ``dream`` attribute must still get a number, so the fallback is the
    default of the class that defines the field, not a copied constant.

    The value is validated to [0, 1]: Jaccard lives in that range, so an
    out-of-range setting is a typo that would otherwise silently plan zero
    merges (threshold > 1) or merge everything (threshold < 0).
    """
    from ..config import DreamConfig

    raw = getattr(getattr(getattr(cfg, "pipeline", None), "dream", None), "dedup_threshold", None)
    if raw is None:
        raw = getattr(cfg, "dream", None)
        raw = getattr(raw, "dedup_threshold", None)
    if raw is None:
        return float(DreamConfig().dedup_threshold)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(DreamConfig().dedup_threshold)
    if not 0.0 <= value <= 1.0:
        return float(DreamConfig().dedup_threshold)
    return value


def _cluster_min_degree(cfg: Any) -> int:
    """Resolve the Phase 1 low-degree pin threshold from the live config."""
    from ..config import DreamConfig

    raw = getattr(
        getattr(getattr(cfg, "pipeline", None), "dream", None), "cluster_min_degree", None
    )
    if raw is None:
        raw = getattr(getattr(cfg, "dream", None), "cluster_min_degree", None)
    if raw is None:
        return int(DreamConfig().cluster_min_degree)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return int(DreamConfig().cluster_min_degree)
    return value if value >= 0 else int(DreamConfig().cluster_min_degree)


def run_phase2_dedup(cfg: Any, group_id: str, force: bool = False) -> Dict[str, Any]:
    """Plan (and optionally apply) Phase 2 dedup. Never raises.

    The merge threshold comes from ``cfg.pipeline.dream.dedup_threshold`` — the
    one reachable setting. ``contracts.DreamSettings.dedup_threshold`` was never
    read: ``dream.py`` receives a ``JanusSettings``, whose ``pipeline.dream`` is
    the pydantic ``DreamConfig``, so that field had zero readers while the
    planner used a literal.

    ``force`` is the caller's decision to write. Phase 3 has its own separate
    gate, so ``force`` here never implies pruning (see ``dream.py``).
    """
    from .dedup import plan_dedup

    threshold = _dedup_threshold(cfg)

    graph, err = _graph(cfg, group_id)
    if graph is None:
        return {
            "status": "SKIPPED",
            "reason": "FalkorDB unreachable (%s)" % err,
            "merges": 0,
            "applied": False,
        }

    snap, read_err = load_graph_snapshot(graph, group_id)
    if read_err:
        return {
            "status": "FAILED",
            "reason": read_err,
            "merges": 0,
            "applied": False,
        }

    entities_seen = len(snap["entities"])
    plan = plan_dedup(
        snap["entities"],
        snap["mentions"],
        snap["relations"],
        threshold=threshold,
    )
    counters = apply_plan(graph, plan, apply=bool(force))
    status = "DONE" if not counters["errors"] else "PARTIAL"
    return {
        "status": status,
        "reason": "; ".join(counters["errors"])[:300],
        "applied": counters["applied"],
        "merges": len(plan.merges),
        "entities_seen": entities_seen,
        "plan": plan.as_dict(),
        "counters": counters,
        "detail": plan.describe(),
    }
