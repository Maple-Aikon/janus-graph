"""Phase 2 must actually converge: MENTIONS endpoints live in TOPOLOGY, not properties.

Measured 2026-10-09 against the live ``graphiti_memory`` graph (26,030 MENTIONS)::

    MATCH (s:Episodic)-[e:MENTIONS]->(v:Entity) RETURN keys(e)
    => ['uuid', 'group_id', 'created_at']   on every single one

``src_uuid``/``dst_uuid`` are not properties that exist on MENTIONS. ``_MENTIONS_Q``
returned ``e.src_uuid, e.dst_uuid``, so both came back NULL on every row and the
failure cascaded:

  * ``plan.mentions_to_redirect`` was structurally 0 -- ``dedup.py:300`` counts
    ``men["dst_uuid"] in redirect`` and ``""`` is never a victim uuid.
  * Phase A read ``e.src_uuid`` -> NULL, then re-CREATEd via
    ``MATCH (:Episodic {uuid: NULL})``. Measured: that pattern matches 0 rows
    (all 5,311 Episodic have a uuid), so both CREATE and DELETE were silent
    no-ops -- yet ``moved`` was incremented, reporting work that never happened.
  * The victim kept its MENTIONS edge, so ``NOT (v)--()`` was false in Phase D
    and the node was never deleted. That is the entire "P2 never converges" report.

RELATES_TO is genuinely unaffected: it really does carry ``source_node_uuid`` /
``target_node_uuid``, both present on every edge.

The fake graph below is faithful on purpose: it resolves each RETURN item by
name, so a query asking for a property MENTIONS does not have gets ``None``
exactly as FalkorDB does. A test that only feeds ``plan_dedup`` a hand-written
dict would have passed on the broken code.
"""

import re

from janus_graph.pipeline.dedup import plan_dedup
from janus_graph.pipeline.dedup_apply import _MENTIONS_Q, apply_plan, load_graph_snapshot

# jaccard(SUMMARY_A, SUMMARY_B) == 0.9231, clear of the 0.8 default. The first
# pair tried scored 0.6923 and yielded a 0-merge plan, which would have made
# these tests pass or fail for a reason unrelated to the bug.
SUMMARY_A = "CRG_TOOLS exposes build embed and refactor tools for the code knowledge graph today"
SUMMARY_B = "CRG_TOOLS exposes build embed and refactor tools for the code knowledge graph"

EPISODIC_UUID = "ep-1"
VICTIM_UUID = "vict"
SURVIVOR_UUID = "surv"

# Keys measured on the real graph. Anything not listed here reads as NULL.
_MENTION_PROPS = {"uuid", "group_id", "created_at"}


class _Result:
    def __init__(self, rows):
        self.result_set = rows


class _FakeFalkor:
    """Stand-in that mirrors the measured edge shape, not an idealised one."""

    def __init__(self):
        self.creates = []
        self.deletes = []
        self.queries = []
        # (src_node_uuid, dst_node_uuid, edge props) for the MENTIONS we care about
        self.mentions = [(EPISODIC_UUID, VICTIM_UUID, {"uuid": "e1", "group_id": "gid"})]
        self.entities = {SURVIVOR_UUID: "gid", VICTIM_UUID: "gid"}
        self.episodics = {EPISODIC_UUID}

    def _resolve(self, item, src, dst, props):
        alias, _, prop = item.strip().partition(".")
        if alias == "e":
            return props.get(prop) if prop in _MENTION_PROPS else None
        if alias in ("s", "a"):
            return src
        if alias in ("n", "v", "b", "t"):
            return dst
        return None

    def query(self, cypher, params=None):
        params = params or {}
        self.queries.append(cypher)

        if "RETURN keys(e)" in cypher:
            return _Result([[sorted(_MENTION_PROPS)]])

        # CREATE must be tested before the generic read branch: the move query
        # now ends in "RETURN count(*)", so a naive `RETURN and MENTIONS` test
        # would swallow it and answer with edge rows instead of a row count.
        if "CREATE" in cypher:
            self.creates.append((cypher, params))
            if "RETURN count(*)" in cypher:
                return _Result([[1]])
            return _Result([])

        if "RETURN" in cypher and "MENTIONS" in cypher:
            wanted = re.search(r"RETURN (.+?)$", cypher.strip(), re.S)
            items = [i.strip() for i in wanted.group(1).split(",")] if wanted else []
            victims = params.get("victims") or []
            rows = []
            for src, dst, props in self.mentions:
                if victims and dst not in victims:
                    continue
                rows.append([self._resolve(i, src, dst, props) for i in items])
            return _Result(rows)

        if "DELETE" in cypher:
            self.deletes.append((cypher, params))
            return _Result([[1]])

        if cypher.strip().startswith("MATCH (n:Entity)"):
            return _Result(
                [[k, "CRG_TOOLS", SUMMARY_A if k == SURVIVOR_UUID else SUMMARY_B, None]
                 for k in (SURVIVOR_UUID, VICTIM_UUID)]
            )

        if "count(" in cypher:
            return _Result([[0]])

        # Unknown edge reads return no rows -- the real code unpacks each row
        # positionally, so returning a short row here would fail the test for a
        # reason that has nothing to do with the bug under test.
        return _Result([])


def _plan():
    return plan_dedup(
        [
            {"uuid": SURVIVOR_UUID, "name": "CRG_TOOLS", "summary": SUMMARY_A,
             "created_at": None},
            {"uuid": VICTIM_UUID, "name": "CRG_TOOLS", "summary": SUMMARY_B,
             "created_at": None},
        ],
        [{"uuid": "e1", "src_uuid": EPISODIC_UUID, "dst_uuid": VICTIM_UUID,
          "created_at": None, "group_id": "gid"}],
        [],
    )


def test_fixture_actually_merges():
    """Guard the fixture itself: a 0-merge plan makes every test below vacuous."""
    plan = _plan()
    assert len(plan.merges) == 1
    assert plan.merges[0].victim_uuid == VICTIM_UUID


def test_mentions_query_reads_endpoints_from_topology_not_properties():
    """``e.src_uuid`` / ``e.dst_uuid`` are not properties MENTIONS has."""
    assert "e.src_uuid" not in _MENTIONS_Q
    assert "e.dst_uuid" not in _MENTIONS_Q
    assert "s.uuid" in _MENTIONS_Q and "n.uuid" in _MENTIONS_Q


def test_load_graph_snapshot_asks_the_database_not_the_dict():
    """End-to-end through the real query path, against a faithful fake."""
    snap, errs = load_graph_snapshot(_FakeFalkor(), "gid")
    assert errs == ""
    assert snap["mentions"][0]["src_uuid"] == EPISODIC_UUID
    assert snap["mentions"][0]["dst_uuid"] == VICTIM_UUID


def test_mentions_to_redirect_is_the_true_count_not_a_structural_zero():
    """dedup.py:300 must see the real destination uuid."""
    snap, _ = load_graph_snapshot(_FakeFalkor(), "gid")
    plan = plan_dedup(
        [
            {"uuid": SURVIVOR_UUID, "name": "CRG_TOOLS", "summary": SUMMARY_A},
            {"uuid": VICTIM_UUID, "name": "CRG_TOOLS", "summary": SUMMARY_B},
        ],
        snap["mentions"],
        [],
    )
    assert len(plan.merges) == 1
    assert plan.mentions_to_redirect == 1


def test_apply_plan_moves_the_mention_and_binds_the_real_episodic():
    """Phase A must really move the edge, and say so."""
    graph = _FakeFalkor()
    counters = apply_plan(graph, _plan(), "gid")

    assert counters["errors"] == []
    assert counters["mentions_moved"] == 1, (
        "before the fix the CREATE/DELETE matched uuid:null and did nothing, "
        "yet moved was still incremented"
    )
    assert graph.creates, "apply_plan must issue the CREATE half of the move"
    payloads = [p for _, p in graph.creates]
    assert any(p.get("src") == EPISODIC_UUID for p in payloads), (
        "CREATE must bind the real Episodic uuid taken from the matched node"
    )
    assert any(p.get("new") == SURVIVOR_UUID for p in payloads)


def test_apply_plan_deletes_the_old_edge_after_creating_the_new_one():
    """Order matters: CREATE first, DELETE second (FalkorDB has no txns)."""
    graph = _FakeFalkor()
    apply_plan(graph, _plan(), "gid")
    assert graph.creates and graph.deletes
    first_delete = next(i for i, q in enumerate(graph.queries) if "DELETE" in q)
    first_create = next(i for i, q in enumerate(graph.queries) if "CREATE" in q)
    assert first_create < first_delete


def test_apply_plan_reports_zero_moved_when_nothing_points_at_victim():
    """Do not report phantom work in the other direction either."""
    graph = _FakeFalkor()
    graph.mentions = [(EPISODIC_UUID, SURVIVOR_UUID, {"uuid": "e9", "group_id": "gid"})]
    counters = apply_plan(graph, _plan(), "gid")
    assert counters["mentions_moved"] == 0
