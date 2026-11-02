"""Phase 2b: byte-identical :Episodic folding.

What these tests pin, and WHY each one exists:

1. THE PLANNER GROUPS ON (name, content) — not on content alone. Measured on
   the 2026-11-02 live graph: 397 content-hash groups carry more than one
   distinct name, so a content-only key would merge across session
   boundaries and lose the distinction the name carries.

2. THE SURVIVOR IS THE EARLIEST created_at (tie-break lowest uuid). Not the
   Entity policy ("longest summary") — for byte-identical content the
   summary length is identical by construction, so that rule would
   degenerate into this one while looking like a different decision.

3. PHASE A MATCHES THE VICTIM ON THE SRC SIDE. The Entity-shaped query
   ``MATCH (s:Episodic)-[e:MENTIONS]->(v:Entity) WHERE v.uuid IN $victims``
   matches ZERO rows for an Episodic victim (measured on the live graph:
   MENTIONS with DST=victim = 0, with SRC=victim = 1 671). A phase built on
   it reports success while moving nothing.

4. PHASE B DROPS THE ROWS THE SURVIVOR ALREADY HAD. Without it the survivor
   points twice at the same Entity.

5. PHASE C DROPS THE DUPLICATES THE MERGE ITSELF CREATED, and only those.
   Two victims in one group pointing at one Entity the survivor did not have
   both land on the survivor.

6. PHASE D DELETES THE VICTIM, AND ONLY A VICTIM THAT IS PROVABLY EMPTY.
   The guard is "no outgoing MENTIONS", measured on the live graph: every
   victim had 0 incoming edges and only MENTIONS outgoing.

7. NO NODE IS DELETED WITHOUT AN UNDO ROW. This is the irreversibility
   guard: the ``dream_entity_undo_log`` table has existed since the schema
   was written and still had 0 rows after 15 dream runs — a table with no
   writer is not an undo path.

8. PHASE 3 REPORT IS NOT A BARE "merged=0". The whole point of the report is
   that an operator can tell "nothing was mergeable" from "nothing was
   looked at".
"""

import re

from janus_graph.pipeline.dedup import (
    DedupPlan,
    EpisodicMerge,
    plan_episodic_exact_dups,
)
from janus_graph.pipeline.dedup_apply import (
    _VICTIM_MENTIONS_Q,
    apply_episodic_plan,
)

GID = "graphiti_memory"

# ── fixtures ─────────────────────────────────────────────────────────────
SURVIVOR = "ep-survivor"
VICTIM = "ep-victim"
VICTIM2 = "ep-victim-2"
ENT_A = "ent-a"
ENT_B = "ent-b"

CONTENT = "the exact same episode body"


class _Result:
    def __init__(self, rows):
        self.result_set = rows


class _FakeGraph:
    """A small graph that answers the queries Phase 2b actually issues.

    Deliberately faithful about the two FalkorDB behaviours that bit this
    work (measured 2026-11-02):

    * a projection whose values repeat DROPS the duplicate row, so a
      2-column ``RETURN s.uuid, t.uuid`` would lose parallel MENTIONS;
    * ``UNWIND $rows ... WHERE e.uuid = row.eu DELETE e`` is a silent NO-OP.
    """

    def __init__(self, mentions=None, episodics=None):
        # mentions: list of (edge_uuid, src_uuid, dst_uuid)
        self.mentions = list(mentions or [])
        self.episodics = dict(episodics or {})
        self.creates = []
        self.deletes = []
        self.queries = []

    def _edge_rows(self, params):
        victims = params.get("victims") or []
        survivors = params.get("survivors") or []
        rows = []
        for eu, src, dst in self.mentions:
            if victims and src not in victims:
                continue
            if survivors and src not in survivors:
                continue
            rows.append([eu, src, dst, "2026-01-01T00:00:00+00:00", GID])
        return rows

    def query(self, cypher, params=None):
        params = params or {}
        self.queries.append(cypher)

        if "CREATE" in cypher:
            self.creates.append((cypher, params))
            # The CREATE really adds the edge to the graph, so the fake's
            # state stays consistent with what FalkorDB would hold.
            self.mentions.append((params["eu"], params["src"], params["dst"]))
            return _Result([[1]])

        if "DELETE" in cypher:
            self.deletes.append((cypher, params))
            if "IN $uus" in cypher:
                uus = set(params.get("uus") or [])
                self.mentions = [m for m in self.mentions if m[0] not in uus]
            else:
                # The endpoint-constrained delete must honour BOTH
                # endpoints. This matters: the rewired edge keeps the OLD
                # edge uuid and sits on the SURVIVOR, so a uuid-only delete
                # would take the freshly created edge instead of the
                # victim's. The production query is safe precisely because
                # it pins the victim's uuid as the source.
                eu = params.get("eu")
                src = params.get("src")
                dst = params.get("dst")
                self.mentions = [
                    m
                    for m in self.mentions
                    if not (
                        m[0] == eu and (src is None or m[1] == src) and (dst is None or m[2] == dst)
                    )
                ]
            return _Result([[1]])

        if "properties(v)" in cypher:
            victims = params.get("victims") or []
            rows = []
            for v in victims:
                if v in self.episodics:
                    rows.append([v, self.episodics[v]])
            return _Result(rows)

        if "RETURN count(e)" in cypher:
            return _Result([[len(self.mentions)]])

        if "RETURN count(v)" in cypher:
            victims = params.get("victims") or []
            return _Result([[len(victims)]])

        if "RETURN" in cypher and "MENTIONS" in cypher:
            rows = self._edge_rows(params)
            return _Result([row[:5] for row in rows])

        return _Result([])


def _merge(victim=VICTIM, survivor=SURVIVOR):
    return [
        EpisodicMerge(
            name="sess",
            survivor_uuid=survivor,
            victim_uuid=victim,
            content_len=len(CONTENT),
        )
    ]


def _graph_with_victim(mentions, victim=VICTIM):
    """A fake graph whose victim node EXISTS.

    The undo path reads the victim's properties, and the delete is blocked
    when the undo rows are short. A victim that does not exist in the graph
    would make every apply-path test fail for the wrong reason.
    """
    return _FakeGraph(
        mentions=mentions,
        episodics={victim: {"uuid": victim, "name": "sess", "content": CONTENT}},
    )


# ── 1. the planner ───────────────────────────────────────────────────────


def test_planner_groups_on_name_and_content_together():
    """Same content, different name => NOT a duplicate.

    The name carries the session/context. Merging across it would lose the
    distinction (measured: 397 content-hash groups carry >1 distinct name).
    """
    nodes = [
        {"uuid": "a", "name": "sess-1", "content": CONTENT, "created_at": "2026-01-01"},
        {"uuid": "b", "name": "sess-2", "content": CONTENT, "created_at": "2026-01-01"},
    ]
    assert plan_episodic_exact_dups(nodes) == []


def test_planner_groups_on_byte_identical_name_and_content():
    nodes = [
        {"uuid": "a", "name": "sess", "content": CONTENT, "created_at": "2026-01-01"},
        {"uuid": "b", "name": "sess", "content": CONTENT, "created_at": "2026-01-02"},
    ]
    merges = plan_episodic_exact_dups(nodes)
    assert len(merges) == 1
    assert {m.victim_uuid for m in merges} == {"b"}
    assert merges[0].survivor_uuid == "a"


def test_planner_survivor_is_earliest_created_at():
    """Not the Entity policy (longest summary) — that would degenerate."""
    nodes = [
        {"uuid": "late", "name": "sess", "content": CONTENT, "created_at": "2026-05-01"},
        {"uuid": "early", "name": "sess", "content": CONTENT, "created_at": "2026-01-01"},
    ]
    merges = plan_episodic_exact_dups(nodes)
    assert merges[0].survivor_uuid == "early"
    assert merges[0].victim_uuid == "late"


def test_planner_tie_break_is_lowest_uuid():
    nodes = [
        {"uuid": "zzz", "name": "sess", "content": CONTENT, "created_at": "2026-01-01"},
        {"uuid": "aaa", "name": "sess", "content": CONTENT, "created_at": "2026-01-01"},
    ]
    merges = plan_episodic_exact_dups(nodes)
    assert merges[0].survivor_uuid == "aaa"


def test_planner_folds_a_group_of_three_into_one_survivor():
    nodes = [
        {"uuid": "a", "name": "s", "content": CONTENT, "created_at": "2026-01-01"},
        {"uuid": "b", "name": "s", "content": CONTENT, "created_at": "2026-01-02"},
        {"uuid": "c", "name": "s", "content": CONTENT, "created_at": "2026-01-03"},
    ]
    merges = plan_episodic_exact_dups(nodes)
    assert len(merges) == 2
    assert {m.survivor_uuid for m in merges} == {"a"}


# ── 3. Phase A matches the victim on the SRC side ───────────────────────


def test_victim_mentions_query_matches_victim_on_the_src_side():
    """The Entity-shaped query matches ZERO Episodic victims."""
    assert "s.uuid IN $victims" in _VICTIM_MENTIONS_Q
    assert "v.uuid IN $victims" not in _VICTIM_MENTIONS_Q
    # The edge uuid MUST be projected: a 2-column projection drops the
    # parallel MENTIONS (measured 2026-11-02: 163 of 33 061 rows).
    assert "e.uuid" in _VICTIM_MENTIONS_Q


# ── 4/5/6. the apply path ────────────────────────────────────────────────


def test_rewire_moves_the_edge_to_the_survivor():
    g = _graph_with_victim([("e1", VICTIM, ENT_A)])
    counters = apply_episodic_plan(
        g,
        _merge(),
        apply=True,
        undo_writer=lambda rows: len(rows),
    )
    assert counters["mentions_rewired"] == 1
    assert counters["mentions_created"] == 1
    assert counters["nodes_deleted"] == 1
    # The CREATE must bind the survivor and the real Entity.
    payloads = [p for _, p in g.creates]
    assert payloads[0]["src"] == SURVIVOR
    assert payloads[0]["dst"] == ENT_A
    # The victim's old edge is gone and the survivor has it.
    assert g.mentions == [("e1", SURVIVOR, ENT_A)]


def test_row_the_survivor_already_has_is_dropped_not_recreated():
    """Phase B. The survivor already mentions ENT_A."""
    g = _graph_with_victim(
        [
            ("e-existing", SURVIVOR, ENT_A),
            ("e1", VICTIM, ENT_A),
        ]
    )
    counters = apply_episodic_plan(
        g,
        _merge(),
        apply=True,
        undo_writer=lambda rows: len(rows),
    )
    assert counters["mentions_dropped"] == 1
    assert counters["mentions_created"] == 0
    # The row was DROPPED, not moved: nothing was created for it.
    assert counters["mentions_moved"] == 0
    # But it IS one of the victim's rows the phase handled.
    assert counters["mentions_rewired"] == 1
    # Every one of the victim's rows is accounted for.
    assert counters["mentions_accounted"] == 1
    # Only the pre-existing row survives.
    assert g.mentions == [("e-existing", SURVIVOR, ENT_A)]


def test_duplicate_created_by_the_merge_is_dropped_separately():
    """Phase C. Two victims, one target the survivor did not have."""
    g = _graph_with_victim(
        [
            ("e1", VICTIM, ENT_B),
            ("e2", VICTIM2, ENT_B),
        ],
        victim=VICTIM,
    )
    g.episodics[VICTIM2] = {"uuid": VICTIM2, "name": "sess", "content": CONTENT}
    merges = [
        EpisodicMerge("s", SURVIVOR, VICTIM, 10),
        EpisodicMerge("s", SURVIVOR, VICTIM2, 10),
    ]
    counters = apply_episodic_plan(
        g,
        merges,
        apply=True,
        undo_writer=lambda rows: len(rows),
    )
    assert counters["survivor_dedupe_dropped"] == 1
    assert counters["mentions_dropped"] == 0
    assert counters["mentions_created"] == 1
    # Exactly one (survivor, ENT_B) row remains.
    assert len([m for m in g.mentions if m[1] == SURVIVOR]) == 1


def test_intra_victim_duplicate_is_counted_not_rewired_twice():
    """The same victim pointing at the same target twice."""
    g = _graph_with_victim(
        [
            ("e1", VICTIM, ENT_A),
            ("e2", VICTIM, ENT_A),
        ]
    )
    counters = apply_episodic_plan(
        g,
        _merge(),
        apply=True,
        undo_writer=lambda rows: len(rows),
    )
    assert counters["mentions_intra_victim_dup"] == 1
    assert counters["mentions_created"] == 1


def test_phase_d_guard_only_deletes_an_empty_victim():
    """A victim that still has a MENTIONS must NOT be deleted."""
    g = _graph_with_victim([("e1", VICTIM, ENT_A)])

    class _StillHasEdge(_FakeGraph):
        def query(self, cypher, params=None):
            if "RETURN count(v)" in cypher:
                return _Result([[0]])
            return super().query(cypher, params)

    g2 = _StillHasEdge(mentions=[("e1", VICTIM, ENT_A)])
    counters = apply_episodic_plan(
        g2,
        _merge(),
        apply=True,
        undo_writer=lambda rows: len(rows),
    )
    assert counters["nodes_deleted"] == 0
    assert counters["victims_not_empty"] == 1


# ── 7. the undo guard ────────────────────────────────────────────────────


def test_no_undo_writer_means_no_node_is_deleted():
    g = _graph_with_victim([("e1", VICTIM, ENT_A)])
    counters = apply_episodic_plan(g, _merge(), apply=True, undo_writer=None)
    assert counters["nodes_deleted"] == 0
    assert counters["errors"]
    assert any("undo" in e for e in counters["errors"])


def test_undo_writer_that_writes_too_few_rows_blocks_the_delete():
    g = _graph_with_victim([("e1", VICTIM, ENT_A)])
    counters = apply_episodic_plan(
        g,
        _merge(),
        apply=True,
        undo_writer=lambda rows: 0,
    )
    assert counters["nodes_deleted"] == 0
    assert any("undo" in e for e in counters["errors"])


def test_undo_rows_capture_the_victim_snapshot_and_its_mentions():
    g = _graph_with_victim([("e1", VICTIM, ENT_A)])
    captured = {}

    def writer(rows):
        captured["rows"] = rows
        return len(rows)

    counters = apply_episodic_plan(g, _merge(), apply=True, undo_writer=writer)
    assert counters["undo_rows_written"] == 1
    rows = captured["rows"]
    assert rows[0]["victim_uuid"] == VICTIM
    assert rows[0]["survivor_uuid"] == SURVIVOR
    assert CONTENT in rows[0]["node_snapshot"]
    assert "e1" in rows[0]["mentions"]


def test_dry_run_writes_nothing_and_still_reports_the_plan():
    g = _graph_with_victim([("e1", VICTIM, ENT_A)])
    counters = apply_episodic_plan(g, _merge(), apply=False)
    assert counters["applied"] is False
    assert counters["mentions_rewired"] == 1
    assert counters["nodes_deleted"] == 0
    assert not g.creates
    assert not g.deletes


# ── 8. the Phase 3 report ────────────────────────────────────────────────


def test_report_explains_a_zero_merge_instead_of_just_stating_it():
    plan = DedupPlan()
    plan.max_observed_similarity = 0.7778
    text = plan.describe()
    assert "0 mergeable Entity pairs at threshold" in text
    assert "0.7778" in text
    # And it always states the Episodic half.
    assert "EPISODIC" in text


def test_report_always_mentions_the_episodic_scan():
    plan = DedupPlan()
    plan.episodic_clusters_scanned = 225
    plan.episodic_duplicate_nodes = 249
    text = plan.describe()
    assert "225" in text
    assert "249" in text
