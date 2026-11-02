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
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

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

#: Read every ``:Episodic`` node for the exact-duplicate planner.
#:
#: ``e.uuid`` is NOT projected here because these are NODES, not edges — but
#: the same projection-dedup trap applies to node properties: two episodes
#: with an identical (name, content) pair project to the SAME tuple, so a
#: 2-column projection would silently collapse them and the planner would see
#: ZERO duplicates. ``created_at`` is projected precisely so the tuple stays
#: distinct per node, and because the survivor policy needs it.
_EPISODIC_Q = (
    "MATCH (n:Episodic) WHERE n.group_id = $gid "
    "RETURN n.uuid, n.name, n.content, n.created_at, n.source_description"
)

#: Read the MENTIONS rows whose SOURCE is one of the given victims.
#:
#: The victim sits on the SRC side. This is the opposite of the Entity
#: phases, where the victim is the DST of a RELATES_TO. Measured on the
#: 2026-11-02 live graph: MENTIONS with DST = victim = 0, MENTIONS with
#: SRC = victim = 1 671. The Entity-shaped query
#: ``MATCH (s:Episodic)-[e:MENTIONS]->(v:Entity) WHERE v.uuid IN $victims``
#: therefore matches NOTHING for an Episodic victim, and a phase built on it
#: reports success while moving zero edges.
#:
#: ``e.uuid`` IS projected: without it, two parallel MENTIONS (same src, same
#: dst, different edge uuid) project to the same tuple and FalkorDB drops one
#: of them. Measured 2026-11-02 on the live graph: the 2-column projection
#: returned 32 898 rows against 33 061 edges — 163 rows lost.
_VICTIM_MENTIONS_Q = (
    "MATCH (s:Episodic)-[e:MENTIONS]->(t:Entity) "
    "WHERE s.uuid IN $victims "
    "RETURN e.uuid, s.uuid, t.uuid, e.created_at, e.group_id"
)

#: Read the MENTIONS rows whose SOURCE is one of the given survivors, used
#: to decide which rewired rows are redundant.
_SURVIVOR_MENTIONS_Q = (
    "MATCH (s:Episodic)-[e:MENTIONS]->(t:Entity) "
    "WHERE s.uuid IN $survivors "
    "RETURN e.uuid, s.uuid, t.uuid, e.created_at, e.group_id"
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
    out: Dict[str, List[Dict]] = {
        "entities": [],
        "mentions": [],
        "relations": [],
        "episodics": [],
    }
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
        for rec in _records(graph, _EPISODIC_Q, {"gid": group_id}):
            rec = (list(rec) + [None] * 5)[:5]
            out["episodics"].append(
                {
                    "uuid": rec[0],
                    "name": rec[1],
                    "content": rec[2],
                    "created_at": str(rec[3]) if rec[3] else None,
                    "source_description": rec[4],
                }
            )
    except Exception as exc:  # noqa: BLE001
        errors.append("episodics: %s" % exc)
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


# ─── Phase 2b: fold byte-identical :Episodic nodes ──────────────────────
#
# WHY A SEPARATE FUNCTION
# -----------------------
# The Entity phases above assume the victim is an ``:Entity`` reached by a
# ``RELATES_TO`` edge, and the victim is the DST of that edge. An Episodic
# victim is the SRC of its MENTIONS edges. Reusing the Entity phases would
# make every Episodic phase match ZERO rows while still reporting success.
#
# The four phases below, in order, and WHY each one exists:
#
#   A. REWIRE — move every MENTIONS row of a victim to its survivor.
#      DELETE + CREATE, because edge endpoints are not writable properties.
#      CREATE happens FIRST: FalkorDB has no transaction, so DELETE-then-
#      CREATE loses the edge outright if the process dies in between.
#
#   B. COLLAPSE — a rewired row whose target the survivor ALREADY mentions
#      is redundant. Kept would make the survivor point twice at the same
#      Entity. This phase DELETES those rows and does not recreate them.
#
#   C. SURVIVOR DEDUPE — after A+B, the survivor can still hold two rows to
#      the SAME target: two victims in one group can point at one Entity the
#      survivor did not have, and both are rewired onto the survivor. Only
#      the rows this merge CREATED are collapsed; redundancy that existed
#      before the merge is left alone (it was not this merge's doing).
#
#   D. DELETE the victim nodes, LAST, and only once provably empty.
#
# ORDER MATTERS. D before A would find 0 empty victims and delete none
# while still reporting success. A before B/C would leave the survivor with
# duplicate edges to the same target.
#
# COUNTERS. Each phase has its own counter. A single "mentions_moved" number
# cannot distinguish "moved the row" from "moved the row AND dropped a
# duplicate" from "moved the row AND created a duplicate" — and those three
# outcomes have different data-loss implications.


def _delete_mentions_by_uuid(graph: Any, edge_uuids: List[str]) -> int:
    """Delete MENTIONS edges by edge uuid; return how many rows went away.

    Uses the ``IN``-list form on purpose. Measured 2026-11-02 against
    FalkorDB 8.6.2:

    * ``MATCH ()-[e:MENTIONS]->() WHERE e.uuid IN $uus DELETE e`` — deletes
      exactly the listed edges, and is fast (500 uuids in <0.01 s).
    * ``UNWIND $rows AS row MATCH (...)-[e:MENTIONS]->(...) WHERE e.uuid =
      row.eu DELETE e`` — a **silent NO-OP**: the edges survive. The
      endpoint-constrained UNWIND delete matches nothing.

    The batch form is used because the per-edge form costs one round trip
    per edge, and this phase moves ~1 700 edges.
    """
    if not edge_uuids:
        return 0
    before = _records(graph, "MATCH ()-[e:MENTIONS]->() RETURN count(e)", {})[0][0]
    _records(
        graph,
        "MATCH ()-[e:MENTIONS]->() WHERE e.uuid IN $uus DELETE e",
        {"uus": list(edge_uuids)},
    )
    after = _records(graph, "MATCH ()-[e:MENTIONS]->() RETURN count(e)", {})[0][0]
    return int(before) - int(after)


def _count_episodic_clusters(episodics: List[Dict[str, Any]]) -> int:
    """Count byte-identical ``:Episodic`` groups (>= 2 members).

    Reported so an operator can tell "there is nothing to merge" from "the
    read never looked at :Episodic nodes". Measured on the 2026-11-02 live
    graph: 225 clusters / 249 victims.
    """
    from .dedup import _content_key

    by_key: Dict[Tuple[str, str], int] = defaultdict(int)
    for node in episodics:
        key = (str(node.get("name") or ""), _content_key(node.get("content")))
        by_key[key] += 1
    return sum(1 for count in by_key.values() if count > 1)


def _build_episodic_undo_rows(
    graph: Any,
    merges: List[Any],
    victim_edges: List[Tuple[str, str, str, Any, Any]],
    group_id: str,
) -> List[Dict[str, Any]]:
    """Capture each victim's full state so the delete can be undone.

    Two things are captured, per victim:

    * ``node_snapshot`` — the victim's complete property map, as JSON. This
      is what a restore needs to recreate the node verbatim.
    * ``mentions`` — the MENTIONS rows that belonged to the victim, as JSON.
      Each row records the edge uuid, the source, the target and whether the
      row was rewired (to whom) or dropped. Without this, restoring the
      node would restore a node with no edges — the memory would come back
      empty.

    The properties are read with ``properties(n)`` rather than a fixed
    column list, because an Episodic node's property set varies (the
    2026-11-02 live graph showed ``entity_edges``, ``entity_types``,
    ``valid_at`` and others, and a fixed list would silently drop the ones
    it did not name).
    """
    import json

    victims = [m.victim_uuid for m in merges]
    if not victims:
        return []

    props_rows = _records(
        graph,
        "MATCH (v:Episodic) WHERE v.uuid IN $victims RETURN v.uuid, properties(v)",
        {"victims": victims},
    )
    props_by_uuid: Dict[str, Dict[str, Any]] = {}
    for rec in props_rows:
        rec = (list(rec) + [None] * 2)[:2]
        if rec[0]:
            props_by_uuid[str(rec[0])] = dict(rec[1] or {})

    edges_by_victim: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for edge_uuid, src_uuid, dst_uuid, created_at, gid in victim_edges:
        edges_by_victim[src_uuid].append(
            {
                "edge_uuid": edge_uuid,
                "src_uuid": src_uuid,
                "dst_uuid": dst_uuid,
                "created_at": str(created_at) if created_at else None,
                "group_id": gid or group_id,
            }
        )

    survivor_of = {m.victim_uuid: m.survivor_uuid for m in merges}
    rows: List[Dict[str, Any]] = []
    for victim_uuid in victims:
        if victim_uuid not in props_by_uuid:
            # The victim node no longer exists — it was deleted by something
            # else, or it was never there. Skip it; the caller compares the
            # row count against the victim count and refuses to delete if
            # they do not match.
            continue
        rows.append(
            {
                "victim_uuid": victim_uuid,
                "survivor_uuid": survivor_of.get(victim_uuid, ""),
                "node_snapshot": json.dumps(props_by_uuid[victim_uuid], default=str),
                "mentions": json.dumps(edges_by_victim.get(victim_uuid, []), default=str),
            }
        )
    return rows


def apply_episodic_plan(
    graph: Any,
    merges: List[Any],
    apply: bool = False,
    batch: int = DEFAULT_BATCH,
    undo_writer: Optional[Any] = None,
    group_id: str = "graphiti_memory",
) -> Dict[str, Any]:
    """Fold byte-identical ``:Episodic`` victims into their survivors.

    ``merges`` is a list of :class:`~janus_graph.pipeline.dedup.EpisodicMerge`
    objects (or any object exposing ``victim_uuid`` / ``survivor_uuid``).

    Nothing is written unless ``apply=True``. The dry-run path reports what
    WOULD happen, using the same measurement logic the apply path uses, so
    the two cannot disagree.

    ``undo_writer`` is an OPTIONAL callable taking the list of undo rows and
    returning how many it persisted. It is a HARD PRECONDITION for deleting
    the victim nodes: if it is missing, raises, or persists fewer rows than
    there are victims, the nodes are NOT deleted and the run reports the
    failure. Deleting a node with no undo record is irreversible, and the
    plan requires the undo path to exist before the delete — the
    ``dream_entity_undo_log`` table proves the point: it has existed since
    the schema was written and still had 0 rows after 15 dream runs.
    """
    counters: Dict[str, Any] = {
        "applied": bool(apply),
        "episodic_merges": len(merges),
        "episodic_victims": 0,
        "mentions_rewired": 0,
        "mentions_moved": 0,
        "mentions_dropped": 0,
        "mentions_created": 0,
        "mentions_intra_victim_dup": 0,
        "survivor_dedupe_dropped": 0,
        "mentions_accounted": 0,
        "nodes_deleted": 0,
        "victims_not_empty": 0,
        "undo_rows_written": 0,
        "errors": [],
    }
    if not merges:
        return counters

    redirect = {m.victim_uuid: m.survivor_uuid for m in merges}
    root = _root_map(redirect)
    victims = list(redirect.keys())
    counters["episodic_victims"] = len(victims)

    # ── Read the victim's MENTIONS rows ─────────────────────────────────
    try:
        ment_rows = _records(
            graph,
            _VICTIM_MENTIONS_Q,
            {"victims": victims},
        )
    except Exception as exc:  # noqa: BLE001
        counters["errors"].append("episodic-mentions-read: %s" % exc)
        return counters

    if not ment_rows:
        counters["errors"].append(
            "episodic-mentions-read: 0 rows for %d victims — refusing to "
            "delete nodes whose edges were never seen" % len(victims)
        )
        return counters

    # ── Read the survivor's MENTIONS rows ───────────────────────────────
    survivors = sorted(set(root.values()))
    try:
        surv_rows = _records(
            graph,
            _SURVIVOR_MENTIONS_Q,
            {"survivors": survivors},
        )
    except Exception as exc:  # noqa: BLE001
        counters["errors"].append("episodic-survivor-read: %s" % exc)
        surv_rows = []

    # The survivor's EXISTING targets: a rewired row whose target is already
    # in this set is redundant and is dropped (phase B), not recreated.
    surv_targets: Dict[str, set] = defaultdict(set)
    for row in surv_rows:
        src_uuid, dst_uuid = str(row[1] or ""), str(row[2] or "")
        if src_uuid and dst_uuid:
            surv_targets[src_uuid].add(dst_uuid)

    # ── Plan the rewire ─────────────────────────────────────────────────
    # ``victim_targets`` maps survivor -> set of targets the victims point
    # at, so phase B can ask "does the survivor already mention this?".
    victim_edges: List[Tuple[str, str, str, Any, Any]] = []
    for edge_uuid, src_uuid, dst_uuid, created_at, gid in ment_rows:
        src_uuid, dst_uuid = str(src_uuid or ""), str(dst_uuid or "")
        if not src_uuid or not dst_uuid:
            counters["errors"].append("episodic-mentions-skip %s: unresolved endpoint" % edge_uuid)
            continue
        victim_edges.append((str(edge_uuid), src_uuid, dst_uuid, created_at, gid))

    # Intra-victim duplicates: the SAME victim pointing at the SAME target
    # twice. These must be collapsed BEFORE the counters are computed, or
    # one logical edge would be counted twice.
    #
    # The redundant row is NOT merely counted — it is DELETED. Leaving it
    # behind would keep a MENTIONS on the victim, and Phase D's guard
    # ("no outgoing MENTIONS") would then refuse to delete the victim: the
    # node survives with a dangling edge, and the run reports a rewrite
    # that left the graph exactly as dirty as before. Measured on the
    # 2026-11-02 live graph the intra-victim count is 0, so this path is
    # currently unexercised by real data — which is exactly why it needs a
    # test.
    seen_victim_pairs: set = set()
    unique_victim_edges: List[Tuple[str, str, str, Any, Any]] = []
    intra_victim_dup_edges: List[str] = []
    for edge in victim_edges:
        key = (edge[1], edge[2])
        if key in seen_victim_pairs:
            counters["mentions_intra_victim_dup"] += 1
            intra_victim_dup_edges.append(edge[0])
            continue
        seen_victim_pairs.add(key)
        unique_victim_edges.append(edge)

    # Which rewired rows are redundant (phase B), and which need a new edge.
    #
    # Two DIFFERENT kinds of redundancy, and they get different counters
    # because they have different causes:
    #
    #   * ``mentions_dropped``         — the survivor ALREADY mentioned this
    #     target before the merge. The row adds nothing.
    #   * ``survivor_dedupe_dropped``  — a SECOND victim in the same group
    #     pointed at the same target, so the first victim's rewire already
    #     gave the survivor that target. This duplicate is CREATED BY the
    #     merge, which is why it is counted separately from pre-existing
    #     redundancy (counting it as "dropped by dedup" would credit the
    #     phase with cleanup it did not perform).
    to_drop: List[str] = []
    survivor_dedupe: List[str] = []
    to_create: List[Tuple[str, str, str, str, Any, Any]] = []
    created_targets: Dict[str, set] = defaultdict(set)
    for edge_uuid, src_uuid, dst_uuid, created_at, gid in unique_victim_edges:
        survivor = root.get(src_uuid, src_uuid)
        if survivor == src_uuid:
            # The victim has no survivor (should not happen) — leave it be.
            counters["errors"].append(
                "episodic-mentions-skip %s: no survivor for %s" % (edge_uuid, src_uuid)
            )
            continue
        if dst_uuid in surv_targets.get(survivor, set()):
            to_drop.append(edge_uuid)
            continue
        if dst_uuid in created_targets.get(survivor, set()):
            survivor_dedupe.append(edge_uuid)
            continue
        created_targets[survivor].add(dst_uuid)
        to_create.append((edge_uuid, src_uuid, survivor, dst_uuid, created_at, gid))

    # The dry-run numbers must be the SAME numbers the apply path will
    # produce — otherwise the dry run is a guess, not a plan. The
    # semantics match the apply path exactly:
    #   mentions_rewired = every victim row the phase handles
    #   mentions_moved   = the rows actually recreated on the survivor
    #   mentions_created = the rows that will exist after the merge
    counters["mentions_rewired"] = len(victim_edges)
    counters["mentions_moved"] = len(to_create)
    counters["mentions_dropped"] = len(to_drop)
    counters["survivor_dedupe_dropped"] = len(survivor_dedupe)
    counters["mentions_created"] = len(to_create)
    counters["mentions_accounted"] = len(victim_edges)

    if not apply:
        counters["nodes_to_delete"] = len(victims)
        counters["mentions_to_delete"] = len(to_drop) + len(survivor_dedupe)
        counters["mentions_to_create"] = len(to_create)
        return counters

    # ── Phase B: drop the redundant rows ────────────────────────────────
    # Deleted BEFORE the rewires are created, so the survivor never holds
    # two rows to the same target even momentarily.
    #
    # The intra-victim duplicates go here too: they are redundant rows on
    # the victim, and they MUST go or the victim keeps an edge and Phase D
    # cannot delete it.
    try:
        _delete_mentions_by_uuid(graph, to_drop + survivor_dedupe + intra_victim_dup_edges)
    except Exception as exc:  # noqa: BLE001
        counters["errors"].append("episodic-mentions-drop: %s" % exc)

    # ── Phase A + C: rewire ─────────────────────────────────────────────
    # CREATE the survivor's edge first, THEN delete the victim's edge.
    # Order matters: FalkorDB has no transaction, so the reverse order
    # destroys the edge outright if the process dies in between.
    rewired = 0
    created = 0
    for edge_uuid, victim_uuid, survivor, dst_uuid, created_at, gid in to_create:
        try:
            made = _records(
                graph,
                "MATCH (s:Episodic {uuid: $src}), (t:Entity {uuid: $dst}) "
                "CREATE (s)-[e:MENTIONS {uuid: $eu, group_id: $gid, "
                "created_at: $ca}]->(t) "
                "RETURN count(*)",
                {
                    "src": survivor,
                    "dst": dst_uuid,
                    "eu": edge_uuid,
                    "gid": gid or "graphiti_memory",
                    "ca": str(created_at) if created_at else None,
                },
            )
            n_made = int(made[0][0]) if made and made[0][0] else 0
            if n_made <= 0:
                counters["errors"].append(
                    "episodic-mentions-create %s: endpoints did not match, edge kept" % edge_uuid
                )
                continue
            created += 1
            # Delete the victim's old edge only after the new one exists.
            #
            # The endpoint constraint matters for two reasons:
            #   1. it removes exactly ONE edge even when parallel MENTIONS
            #      share (src, dst) — measured 2026-11-02;
            #   2. the newly created edge carries the SAME e.uuid but sits on
            #      the SURVIVOR, so the victim-side endpoint constraint
            #      cannot match it. (FalkorDB does not enforce edge-uuid
            #      uniqueness — measured — so the two edges briefly share a
            #      uuid, and only the endpoint constraint keeps the DELETE
            #      from taking both.)
            _records(
                graph,
                "MATCH (:Episodic {uuid: $src})-[e:MENTIONS]->(:Entity {uuid: $dst}) "
                "WHERE e.uuid = $eu DELETE e",
                {"src": victim_uuid, "dst": dst_uuid, "eu": edge_uuid},
            )
            rewired += 1
        except Exception as exc:  # noqa: BLE001
            counters["errors"].append("episodic-mentions-move %s: %s" % (edge_uuid, exc))
    counters["mentions_rewired"] = rewired
    counters["mentions_created"] = created

    # ``mentions_rewired`` is the number of the VICTIM's MENTIONS rows the
    # phase handled — every row it removed from the victim, whether it was
    # recreated on the survivor or dropped as redundant.
    #
    # This is the semantics the plan's acceptance formula is written in:
    #     MENTIONS_after = MENTIONS_before - mentions_rewired
    #                                     + mentions_created
    #
    # The plan's older formula subtracted ``survivor_dedupe`` as well. That
    # was written for a design that rewire FIRST and then deleted the
    # duplicates the rewire had just created on the survivor — so the
    # survivor-side duplicates were rows that never came from the victim.
    # This implementation collapses BEFORE the rewire (a survivor never
    # holds two rows to the same target even momentarily), so the
    # merge-introduced duplicates are rows OF THE VICTIM: they are part of
    # ``mentions_rewired``, not an extra subtraction. Measured on the clone
    # 2026-11-02: 1 671 victim rows, 441 dropped, 22 intra-victim, 38
    # merge-introduced, 1 170 recreated — and the observed edge delta
    # (33 064 -> 32 563 = -501) equals ``created - rewired`` exactly.
    #
    # ``mentions_moved`` keeps the narrower number (the rows actually
    # recreated on the survivor) so both readings stay available.
    counters["mentions_moved"] = rewired
    counters["mentions_rewired"] = len(victim_edges)
    counters["mentions_accounted"] = len(victim_edges)

    # ── UNDO: record every victim BEFORE deleting any node ──────────────
    # The victims' full property maps and the MENTIONS rows that were
    # rewired are captured NOW, while the victims still exist. A backup
    # taken after the delete would be a backup of nothing.
    undo_rows: List[Dict[str, Any]] = []
    try:
        undo_rows = _build_episodic_undo_rows(graph, merges, unique_victim_edges, group_id)
    except Exception as exc:  # noqa: BLE001
        counters["errors"].append("episodic-undo-build: %s" % exc)
        undo_rows = []

    if len(undo_rows) < len(victims):
        # Refuse to delete. The undo path is incomplete, so a delete would be
        # irreversible for at least one victim.
        counters["errors"].append(
            "episodic-undo: only %d undo rows for %d victims — nodes NOT deleted"
            % (len(undo_rows), len(victims))
        )
        counters["victims_not_empty"] = len(victims)
        return counters

    if undo_writer is None:
        counters["errors"].append("episodic-undo: no undo_writer supplied — nodes NOT deleted")
        counters["victims_not_empty"] = len(victims)
        return counters

    try:
        written = int(undo_writer(undo_rows) or 0)
    except Exception as exc:  # noqa: BLE001
        counters["errors"].append("episodic-undo-write: %s" % exc)
        written = 0
    counters["undo_rows_written"] = written
    if written < len(victims):
        counters["errors"].append(
            "episodic-undo: wrote %d of %d undo rows — nodes NOT deleted" % (written, len(victims))
        )
        counters["victims_not_empty"] = len(victims)
        return counters

    # ── Phase D: delete the emptied victim nodes ────────────────────────
    # Guard: the victim must have NO outgoing MENTIONS left. The stricter
    # `NOT (v)--()` is not used: it would also refuse to delete a victim
    # that still carries some other edge type, and the plan's measured
    # invariant is about MENTIONS specifically. Measured 2026-11-02 on the
    # live graph: every victim had 0 incoming edges and only MENTIONS
    # outgoing, so this guard empties 100 % of the victims.
    deleted = 0
    not_empty = 0
    for start in range(0, len(victims), batch):
        chunk = victims[start : start + batch]
        try:
            res = _records(
                graph,
                "MATCH (v:Episodic) WHERE v.uuid IN $victims "
                "AND NOT (v)-[:MENTIONS]->() RETURN count(v)",
                {"victims": chunk},
            )
            n_before = int(res[0][0]) if res and res[0][0] is not None else 0
            graph.query(
                "MATCH (v:Episodic) WHERE v.uuid IN $victims "
                "AND NOT (v)-[:MENTIONS]->() DETACH DELETE v",
                {"victims": chunk},
            )
            deleted += n_before
            not_empty += len(chunk) - n_before
        except Exception as exc:  # noqa: BLE001
            counters["errors"].append("episodic-node-delete: %s" % exc)
    counters["nodes_deleted"] = deleted
    counters["victims_not_empty"] = not_empty

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


def run_phase2_dedup(
    cfg: Any,
    group_id: str,
    force: bool = False,
    queue: Optional[Any] = None,
    run_id: str = "",
) -> Dict[str, Any]:
    """Plan (and optionally apply) Phase 2 dedup. Never raises.

    The merge threshold comes from ``cfg.pipeline.dream.dedup_threshold`` — the
    one reachable setting. ``contracts.DreamSettings.dedup_threshold`` was never
    read: ``dream.py`` receives a ``JanusSettings``, whose ``pipeline.dream`` is
    the pydantic ``DreamConfig``, so that field had zero readers while the
    planner used a literal.

    ``force`` is the caller's decision to write. Phase 3 has its own separate
    gate, so ``force`` here never implies pruning (see ``dream.py``).
    """
    from .dedup import plan_dedup, plan_episodic_exact_dups

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

    # ── Phase 2b: byte-identical :Episodic folds ───────────────────────
    # Separate from the Entity phases above: the victim is an :Episodic node
    # and the survivor policy is different (earliest created_at, not
    # "longest summary"). The Entity guard ``NOT (v)--()`` would also match
    # zero Episodic victims, because their edges are MENTIONS to Entities.
    episodic_plan = plan_episodic_exact_dups(snap["episodics"])
    plan.episodic_clusters_scanned = _count_episodic_clusters(snap["episodics"])
    plan.episodic_duplicate_nodes = len(episodic_plan)

    # The undo writer is the HARD PRECONDITION for deleting victim nodes.
    # Without it the Episodic phase still rewires the MENTIONS (that part is
    # reversible — the edges keep their identity), but it will NOT delete the
    # victim nodes, and it says so in ``errors``.
    undo_writer = None
    if queue is not None and run_id:

        def undo_writer(rows):  # noqa: ANN001 - tiny sync shim
            return queue.record_episodic_undo_sync(run_id, rows)

    episodic_counters = apply_episodic_plan(
        graph,
        episodic_plan,
        apply=bool(force),
        undo_writer=undo_writer,
        group_id=group_id,
    )

    status = "DONE" if not counters["errors"] else "PARTIAL"
    if episodic_counters["errors"]:
        status = "PARTIAL"
    return {
        "status": status,
        "reason": "; ".join(counters["errors"] + episodic_counters["errors"])[:300],
        "applied": counters["applied"],
        "merges": len(plan.merges),
        "episodic_merges": len(episodic_plan),
        "entities_seen": entities_seen,
        "episodics_seen": len(snap["episodics"]),
        "plan": plan.as_dict(),
        "counters": counters,
        "episodic_counters": episodic_counters,
        "detail": plan.describe(),
    }
