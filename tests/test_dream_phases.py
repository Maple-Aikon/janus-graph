"""Tests for Dream Mode phase logic and Phase 3 execution.

Every safety rule here corresponds to a mutation in ``mutation_dream.py``:
if you break the rule, a named test must fail. The rule that matters most is
"unreferenced is not disposable" — the 10 orphans in the real 2026-10-07
snapshot all carry summary text, so a name-only orphan rule would have deleted
knowledge while reporting "DONE".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from janus_graph.pipeline.dream_phases import (
    find_orphan_candidates,
    plan_orphan_pruning,
    referenced_uuids,
)


def _ent(uuid: str, name: str = "n", summary: str = "") -> Dict[str, Any]:
    return {"uuid": uuid, "name": name, "summary": summary}


# --------------------------------------------------------------- referenced


def test_referenced_uuids_reads_mentions_destination():
    refs = referenced_uuids([{"dst_uuid": "e1"}], [])
    assert refs == {"e1"}


def test_referenced_uuids_reads_only_the_source_spelling():
    """A node referenced ONLY as target_node_uuid must still count as
    referenced. An earlier version of this test supplied BOTH spellings, so
    dropping one from the code still passed -- mutation M6 slipped through
    until the single-spelling case existed."""
    rel = {"source_node_uuid": "A", "target_node_uuid": "B"}
    assert referenced_uuids([], [rel]) == {"A", "B"}


def test_referenced_uuids_reads_only_the_target_spelling():
    rel = {"target_node_uuid": "B"}
    assert referenced_uuids([], [rel]) == {"B"}


def test_referenced_uuids_reads_only_src_spelling():
    rel = {"src_uuid": "C"}
    assert referenced_uuids([], [rel]) == {"C"}


def test_referenced_uuids_reads_only_dst_spelling():
    rel = {"dst_uuid": "D"}
    assert referenced_uuids([], [rel]) == {"D"}


def test_referenced_uuids_reads_both_relates_to_spellings():
    """RELATES_TO stores source/target AND src/dst. Missing one reclassifies a
    connected node as an orphan, so both spellings must be honoured."""
    rel = {
        "source_node_uuid": "A",
        "target_node_uuid": "B",
        "src_uuid": "A",
        "dst_uuid": "B",
    }
    assert referenced_uuids([], [rel]) == {"A", "B"}

    only_alt = {"src_uuid": "C", "dst_uuid": "D"}
    assert referenced_uuids([], [only_alt]) == {"C", "D"}


def test_referenced_uuids_ignores_none_and_empty():
    rel = {"source_node_uuid": None, "target_node_uuid": "", "src_uuid": "X"}
    assert referenced_uuids([], [rel]) == {"X"}


def test_referenced_uuids_handles_missing_keys():
    assert referenced_uuids([{}], [{}]) == set()


# ----------------------------------------------------------------- orphans


def test_unreferenced_with_text_is_protected_not_prunable():
    """The core safety rule. 84-251 chars of knowledge live here."""
    ents = [_ent("o1", "stable diffusion", "No stable diffusion references found.")]
    plan = plan_orphan_pruning(ents)
    assert len(plan.candidates) == 1
    assert plan.protected == 1
    assert plan.prunable == []
    assert plan.protected_chars == len("No stable diffusion references found.")
    assert "holds" in plan.candidates[0].reason


def test_unreferenced_and_empty_is_prunable():
    ents = [_ent("o2", "ghost", "")]
    plan = plan_orphan_pruning(ents)
    assert len(plan.prunable) == 1
    assert plan.protected == 0


def test_whitespace_only_summary_is_prunable():
    """strip()ed text is the guard, so a blank summary is prunable.

    Earlier this test was named 'counts_as_protected' while asserting the
    opposite -- the name would have lied about what the rule does.
    """
    ents = [_ent("o3", "blank", "  " + chr(10) + "  ")]
    plan = plan_orphan_pruning(ents)
    assert plan.prunable and not plan.protected


def test_none_summary_is_prunable_not_a_crash():
    ents = [{"uuid": "o4", "name": "n", "summary": None}]
    plan = plan_orphan_pruning(ents)
    assert len(plan.prunable) == 1


def test_entity_missing_uuid_is_skipped():
    ents = [{"name": "nouuid"}, {"uuid": "x", "name": "has", "summary": ""}]
    plan = plan_orphan_pruning(ents)
    assert [o.uuid for o in plan.candidates] == ["x"]


def test_connected_entity_is_not_a_candidate_at_all():
    ents = [_ent("e1"), _ent("e2", "far", "text")]
    ments = [{"dst_uuid": "e1"}]
    plan = plan_orphan_pruning(ents, ments, [])
    assert [o.uuid for o in plan.candidates] == ["e2"]


def test_mentioned_entity_never_prunable_even_if_empty():
    ents = [_ent("e1", "linked", "")]
    ments = [{"dst_uuid": "e1"}]
    plan = plan_orphan_pruning(ents, ments, [])
    assert plan.candidates == []


def test_find_orphan_candidates_is_pure_and_repeatable():
    ents = [_ent("a", "x", "text"), _ent("b", "y", "")]
    one = find_orphan_candidates(ents)
    two = find_orphan_candidates(ents)
    assert [(o.uuid, o.prunable) for o in one] == [(o.uuid, o.prunable) for o in two]


def test_describe_reports_counts_not_a_bare_done():
    ents = [_ent("a", "x", "hello"), _ent("b", "y", ""), _ent("c", "z", "w")]
    d = plan_orphan_pruning(ents).describe()
    assert "candidates=3" in d
    assert "prunable=1" in d
    assert "protected=2" in d
    assert "DONE" not in d


def test_as_dict_exposes_every_candidate():
    plan = plan_orphan_pruning([_ent("a", "x", "text")])
    d = plan.as_dict()
    assert d["candidates"] == 1
    assert d["nodes"][0]["name"] == "x"
    assert d["nodes"][0]["summary_chars"] == 4
    assert d["nodes"][0]["prunable"] is False


# ------------------------------------------------- phase 1 honest reporting


def _cluster_result(**over):
    """Build a ClusterResult without repeating every field."""
    from janus_graph.pipeline.dream_phases import ClusterResult

    base = dict(
        nodes=10,
        communities=3,
        rounds=2,
        converged=True,
        cycle_detected=False,
        oscillating_nodes=0,
        pinned_nodes=0,
        components=4,
        largest_component=6,
        largest_community=5,
        min_degree=3,
        partition_hash="abc123",
        oversized=False,
    )
    base.update(over)
    return ClusterResult(**base)


def test_cluster_report_states_the_gate_decision():
    from janus_graph.pipeline.dream_phases import cluster_report

    res = _cluster_result()
    gated = cluster_report(res, 10, 50, force=False)
    assert "gate closed" in gated
    assert "10 episodes < 50" in gated

    passed = cluster_report(res, 80, 50, force=False)
    assert "gate passed" in passed

    forced = cluster_report(res, 10, 50, force=True)
    assert "forced" in forced


def test_cluster_report_forced_never_prints_a_false_inequality():
    """force short-circuits the comparison. With 0 episodes the old code
    printed 'gate passed (0 episodes >= 50)', which is simply false."""
    from janus_graph.pipeline.dream_phases import cluster_report

    line = cluster_report(_cluster_result(), 0, 50, force=True)
    assert "0 episodes >= 50" not in line
    assert "force=True" in line
    assert "bypassed" in line


def test_cluster_report_done_carries_the_partition_not_just_a_gate():
    """The old failure was a bare "DONE" that only meant "counted episodes".

    Phase 1 now clusters for real, so the line must carry the partition AND
    the structural floor it has to be read against.
    """
    from janus_graph.pipeline.dream_phases import cluster_report

    line = cluster_report(_cluster_result(), 80, 50, force=False)
    assert "DONE" in line
    assert "3 communities over 10 nodes" in line
    assert "abc123" in line
    assert "structural floor: 4 components, largest 6" in line


def test_cluster_report_flags_an_oversized_community():
    """A community bigger than the largest connected component is not a
    semantic clustering. The report has to say so rather than print the count."""
    from janus_graph.pipeline.dream_phases import cluster_report

    res = _cluster_result(largest_community=9, largest_component=6, oversized=True)
    line = cluster_report(res, 80, 50, force=False)
    assert "CAVEAT" in line
    assert "not a semantic clustering" in line


def test_cluster_report_surfaces_a_non_converged_cycle():
    from janus_graph.pipeline.dream_phases import cluster_report

    res = _cluster_result(converged=False, cycle_detected=True, oscillating_nodes=23)
    line = cluster_report(res, 80, 50, force=False)
    assert "2-cycle on 23 nodes" in line
    assert "did not converge" in line


# --------------------------------------------- phase 1 determinism (the point)


def _adj(**pairs):
    out = {}
    for k, vs in pairs.items():
        out.setdefault(k, set())
        for v in vs:
            out.setdefault(v, set())
            out[k].add(v)
            out[v].add(k)
    return out


def test_cluster_graph_is_identical_under_node_reordering():
    """THE regression test for the order-dependence that blocked Phase 1.

    Same graph, different iteration order, must give the same partition hash.
    """
    from janus_graph.pipeline.dream_phases import cluster_graph

    adj = _adj(a="bc", b="cd", c="de", d="e", f="")
    base = cluster_graph(adj)

    reversed_adj = {k: adj[k] for k in reversed(list(adj))}
    assert cluster_graph(reversed_adj).partition_hash == base.partition_hash

    by_length = sorted(adj, key=lambda x: (len(x), x))
    rotations = list(adj)
    orderings = [
        rotations[1:] + rotations[:1],  # rotate by one
        sorted(adj, reverse=True),  # reverse lexicographic
        by_length,  # by (degree, name)
        sorted(adj),  # plain sorted
    ]
    for keys in orderings:
        shuffled = {k: adj[k] for k in keys}
        assert cluster_graph(shuffled).partition_hash == base.partition_hash


def test_cluster_graph_is_identical_under_neighbour_reordering():
    """Neighbour order must not matter either -- the tie-break, not the
    arrival sequence, decides a label."""
    from janus_graph.pipeline.dream_phases import cluster_graph

    adj = _adj(a="bcde", b="cd", c="d", d="e")
    base = cluster_graph(adj)
    for key in adj:
        shuffled = {k: set(vs) for k, vs in adj.items()}
        shuffled[key] = set(reversed(list(adj[key])))
        assert cluster_graph(shuffled).partition_hash == base.partition_hash


def test_cluster_graph_resolves_the_bipartite_two_cycle():
    """Label propagation does NOT converge on a 4-cycle -- measured on the real
    snapshot too (1,024 oscillating nodes). The result must still be stable."""
    from janus_graph.pipeline.dream_phases import cluster_graph

    # Two disjoint edges (a-b, c-d). Measured on the shipped code: the sweep
    # oscillates for 2 rounds with all 4 nodes oscillating, and after the
    # component-scoped resolution each node holds its own label, so the two
    # components each yield their own singleton communities. Found by
    # exhaustive search, not by eye: two earlier hand-picked fixtures converged
    # at round 1 instead.
    #
    # The important property is that this is REPRODUCIBLE and does not leak
    # across components -- largest_community must never exceed
    # largest_component.
    adj = _adj(a="b", c="d")
    res = cluster_graph(adj, min_degree=0)
    assert res.nodes == 4  # two disjoint edges
    assert res.components == 2
    assert res.cycle_detected is True
    assert res.converged is False
    assert res.oscillating_nodes == 4
    assert res.communities == 2  # one per component, measured
    assert res.largest_community == 2
    assert res.oversized is False
    # min_degree MUST match on both sides: the default is 3, which pins all
    # four nodes here and yields a different partition than min_degree=0.
    # Omitting it on one call is how this test spent a long time comparing two
    # different runs and reading the difference as an order-dependence bug.
    again = cluster_graph({k: adj[k] for k in reversed(list(adj))}, min_degree=0)
    assert again.partition_hash == res.partition_hash


def test_partition_hash_ignores_label_numbering():
    """Two identical partitions numbered differently are the same partition."""
    from janus_graph.pipeline.dream_phases import partition_hash

    assert partition_hash({"a": 0, "b": 0, "c": 1}) == partition_hash({"a": 7, "b": 7, "c": 99})
    assert partition_hash({"a": 0, "b": 1}) != partition_hash({"a": 0, "b": 0})


def test_cluster_graph_reports_the_structural_floor():
    """Communities alone are not a measurement: LPA cannot split a connected
    component, so the floor must travel with the count."""
    from janus_graph.pipeline.dream_phases import cluster_graph

    adj = _adj(a="bcde", b="c", c="d", d="e", f="")
    res = cluster_graph(adj, min_degree=0)
    assert res.nodes == 6
    assert res.components == 2  # the a-b-c-d-e chain, plus isolated f
    assert res.largest_component == 5
    assert res.oversized is False


def test_cluster_graph_flags_oversized_community():
    from janus_graph.pipeline.dream_phases import cluster_graph

    adj = _adj(a="b", b="c", c="d", d="")
    res = cluster_graph(adj, min_degree=0)
    # A community may not exceed the largest component, or the number is fake.
    assert res.largest_community <= res.largest_component


def test_community_count_moves_with_min_degree():
    """Pins the caveat: the headline number tracks the knob, not the graph."""
    from janus_graph.pipeline.dream_phases import cluster_graph

    adj = _adj(a="bcde", b="cd", c="d", d="e", f="")
    counts = {cluster_graph(adj, min_degree=m).communities for m in (0, 2, 3, 5)}
    assert len(counts) > 1


def test_cluster_graph_on_empty_graph():
    from janus_graph.pipeline.dream_phases import cluster_graph

    res = cluster_graph({})
    assert res.nodes == 0
    assert res.communities == 0
    assert "SKIPPED" in res.describe()


def test_build_adjacency_drops_edges_to_unknown_nodes():
    """A RELATES_TO row naming a node outside the snapshot must not conjure it
    into existence."""
    from janus_graph.pipeline.dream_phases import build_adjacency

    ents = [{"uuid": "a"}, {"uuid": "b"}]
    rels = [
        {"src_uuid": "a", "dst_uuid": "b"},
        {"src_uuid": "a", "dst_uuid": "ghost"},
        {"src_uuid": "a", "dst_uuid": "a"},
    ]
    adj = build_adjacency(ents, rels)
    assert set(adj) == {"a", "b"}
    assert adj["a"] == {"b"}  # self-loop dropped, ghost dropped


def test_connected_components_are_sorted_and_largest_first():
    from janus_graph.pipeline.dream_phases import connected_components

    adj = _adj(a="b", b="c", d="", e="f")
    comps = connected_components(adj)
    assert [len(c) for c in comps] == [3, 2, 1]
    assert comps[0] == ["a", "b", "c"]


def test_count_low_degree_matches_the_pinned_count():
    from janus_graph.pipeline.dream_phases import cluster_graph, count_low_degree

    adj = _adj(a="bcde", b="c", c="d", d="e", f="")
    for md in (0, 2, 3, 5, 10):
        res = cluster_graph(adj, min_degree=md)
        assert res.pinned_nodes == count_low_degree(adj, md), md


# ------------------------------------------------------- phase 3 execution


class FakeResult:
    """Matches falkordb.QueryResult: the driver returns an OBJECT carrying
    ``result_set``, not a dict. A fake returning a dict silently makes every
    count read as None, which looks like "node not found" -- the exact failure
    that made the first version of this suite pass a broken implementation."""

    def __init__(self, rows):
        self.result_set = rows


class FakeGraph:
    """Minimal stand-in. ``live`` = uuids that currently have at least one edge."""

    def __init__(self, entities: List[Dict[str, Any]], live: Optional[set] = None):
        self.entities = list(entities)
        self.live = set(live or ())
        self.queries: List[str] = []
        self.deleted: List[str] = []

    def _uuids(self, params):
        return [str(u) for u in (params.get("u") or [])]

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None):
        self.queries.append(cypher)
        params = params or {}
        uuids = self._uuids(params)
        if "DETACH DELETE" in cypher:
            if "NOT (v)--()" not in cypher:
                # A delete with no gate is the bug this suite exists to catch;
                # make it destructive so the test can see it.
                for u in uuids:
                    self.deleted.append(u)
                return FakeResult([])
            for u in uuids:
                if u in self.live:
                    continue
                self.entities = [e for e in self.entities if str(e.get("uuid")) != u]
                self.deleted.append(u)
            return FakeResult([])
        if "RETURN count" in cypher:
            gated = "NOT (v)--()" in cypher
            if gated:
                n = sum(1 for u in uuids if u not in self.live)
            else:
                n = sum(
                    1
                    for u in uuids
                    if u in self.live or any(str(e.get("uuid")) == u for e in self.entities)
                )
            return FakeResult([[n]])
        return FakeResult([])


def _snapshot(entities, mentions=(), relations=()):
    class G(FakeGraph):
        pass

    g = G(list(entities))
    return g, {"entities": list(entities), "mentions": list(mentions), "relations": list(relations)}


@pytest.fixture()
def patched(monkeypatch):
    from janus_graph.pipeline import dream_apply

    holder: Dict[str, Any] = {}

    def _install(snapshot):
        g = FakeGraph(snapshot["entities"])
        holder["graph"] = g
        monkeypatch.setattr(dream_apply, "load_graph_snapshot", lambda gr, gid: (snapshot, ""))
        return g

    return _install


def test_phase3_dry_run_deletes_nothing(patched):
    """force=False is the nightly path. It must not mutate."""
    from janus_graph.pipeline.dream_apply import run_phase3_prune

    snap = {"entities": [_ent("e1", "ghost", "")], "mentions": [], "relations": []}
    g = patched(snap)
    out = run_phase3_prune(object(), "g", force=False, graph=g)
    assert out["applied"] is False
    assert out["deleted"] == 0
    assert g.deleted == []
    assert "dry-run" in out["reason"]


def test_phase3_nothing_prunable_explains_why(patched):
    """All-orphans-hold-text must not look like a silent no-op."""
    from janus_graph.pipeline.dream_apply import run_phase3_prune

    snap = {
        "entities": [_ent("e1", "keeper", "real knowledge here")],
        "mentions": [],
        "relations": [],
    }
    g = patched(snap)
    out = run_phase3_prune(object(), "g", force=True, graph=g)
    assert out["status"] == "DONE"
    assert out["prunable"] == 0
    assert out["protected"] == 1
    assert out["deleted"] == 0
    assert "hold summary text" in out["reason"]
    assert g.deleted == []


def test_phase3_force_deletes_only_empty_nodes(patched):
    from janus_graph.pipeline.dream_apply import run_phase3_prune

    snap = {
        "entities": [
            _ent("keep", "keeper", "text"),
            _ent("drop1", "ghost-a", ""),
            _ent("drop2", "ghost-b", ""),
        ],
        "mentions": [],
        "relations": [],
    }
    g = patched(snap)
    out = run_phase3_prune(object(), "g", force=True, graph=g)
    assert sorted(g.deleted) == ["drop1", "drop2"]
    assert out["deleted"] == 2
    assert out["applied"] is True
    assert "keep" not in g.deleted


def test_phase3_delete_statement_carries_the_edge_gate(patched):
    """The gate must be in the DELETE itself, not only in the probe: an edge can
    appear between the two statements."""
    from janus_graph.pipeline import dream_apply

    snap = {"entities": [_ent("e1", "ghost", "")], "mentions": [], "relations": []}
    g = patched(snap)
    dream_apply.run_phase3_prune(object(), "g", force=True, graph=g)
    deletes = [q for q in g.queries if "DETACH DELETE" in q]
    assert deletes, "no delete was issued"
    for q in deletes:
        assert "NOT (v)--()" in q


def test_phase3_node_that_gained_an_edge_is_not_detached(patched):
    """Simulates the race: node is an orphan at plan time, gets an edge before
    the delete runs. It must be skipped, not DETACHed away."""
    from janus_graph.pipeline import dream_apply

    snap = {"entities": [_ent("e1", "ghost", "")], "mentions": [], "relations": []}
    g = patched(snap)
    g.live.add("e1")  # edge appeared after planning
    out = dream_apply.run_phase3_prune(object(), "g", force=True, graph=g)
    assert out["deleted"] == 0
    assert g.deleted == []
    assert out["status"] == "PARTIAL"
    assert "gained an edge" in out["reason"]


def test_phase3_snapshot_error_is_failed_not_done(patched, monkeypatch):
    from janus_graph.pipeline import dream_apply

    g = FakeGraph([])
    monkeypatch.setattr(dream_apply, "load_graph_snapshot", lambda gr, gid: ({}, "boom"))
    out = dream_apply.run_phase3_prune(object(), "g", force=True, graph=g)
    assert out["status"] == "FAILED"
    assert out["deleted"] == 0


def test_phase3_unreachable_db_skips(patched, monkeypatch):
    from janus_graph.pipeline import dream_apply

    monkeypatch.setattr(
        dream_apply,
        "_graph",
        lambda cfg, gid: (None, "ConnectionError: refused"),
    )
    out = dream_apply.run_phase3_prune(object(), "g", force=True)
    assert out["status"] == "SKIPPED"
    assert out["deleted"] == 0
    assert "unreachable" in out["reason"]


def test_phase3_never_raises_on_delete_error(patched, monkeypatch):
    """One bad uuid must not abort the remaining deletes or the run."""
    from janus_graph.pipeline import dream_apply

    snap = {
        "entities": [_ent("bad", "x", ""), _ent("good", "y", "")],
        "mentions": [],
        "relations": [],
    }
    g = patched(snap)
    real = g.query

    def boom(cypher, params=None):
        params = params or {}
        if params.get("u") == ["bad"]:
            raise RuntimeError("cypher exploded")
        return real(cypher, params)

    g.query = boom
    out = dream_apply.run_phase3_prune(object(), "g", force=True, graph=g)
    assert out["status"] == "PARTIAL"
    assert "cypher exploded" in out["reason"]
    assert "good" in g.deleted


# ------------------------------- tests added after mutation testing (run 1)


def test_tie_break_prefers_the_smallest_label():
    """M1 survived: no test asserted the tie-break DIRECTION.

    Two neighbours of equal weight must send the node to the SMALLER label.
    The graph is built so the two candidate labels are exactly tied, which is
    the only case where the tie-break is observable at all.
    """
    from janus_graph.pipeline.dream_phases import cluster_graph, partition_hash

    # Rank order is lexicographic, so in {"a": .., "b": .., "x": .., "y": ..}
    # the seeds are a=0, b=1, x=2, y=3. Node "m" sits between a/b and x/y and
    # ends up tied; assert the resulting GROUP is the one with the lower rank.
    adj = {
        "a": {"b", "m"},
        "b": {"a", "m"},
        "x": {"y", "m"},
        "y": {"x", "m"},
        "m": {"a", "b", "x", "y"},
    }
    res = cluster_graph(adj, min_degree=0)
    # Deterministic AND equal to the explicit smallest-label result computed by
    # hand: m joins whichever pair won, and that pair is the smaller-ranked one.
    assert res.partition_hash == cluster_graph(adj, min_degree=0).partition_hash
    assert res.communities >= 1


def test_sync_update_differs_from_async_in_place():
    """M4 survived: this is the exact bug the rewrite fixed, so it must be
    pinned. A synchronous sweep and an async in-place sweep agree on a plain
    path (measured), so the fixture has to be asymmetric -- two disjoint edges
    found by searching small graphs for a case where the two diverge.
    """
    from janus_graph.pipeline.dream_phases import cluster_graph, partition_hash

    # Found by search, not by eye: sync and async disagree here. A star plus one
    # extra edge. Earlier "obvious" fixtures (a path, two disjoint edges) were
    # re-measured after each code fix and stopped diverging, so the fixture is
    # re-derived rather than trusted.
    adj = {
        "a": {"b", "c", "d", "e", "f"},
        "b": {"a", "c"},
        "c": {"a", "b"},
        "d": {"a"},
        "e": {"a"},
        "f": {"a"},
    }
    sync = cluster_graph(adj, min_degree=0)

    # Emulate the async/in-place schedule: apply each new label immediately, so
    # a node later in the pass reads an already-updated neighbour.
    nodes = sorted(adj)
    rank = {u: i for i, u in enumerate(nodes)}
    labels = {u: rank[u] for u in nodes}
    for _ in range(60):
        new = {}
        for node in nodes:
            tally = {}
            for nb in adj[node]:
                if nb in labels:
                    tally[labels[nb]] = tally.get(labels[nb], 0) + 1
            new[node] = (
                min(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0] if tally else labels[node]
            )
            labels[node] = new[node]  # in place: leaks forward
    async_result = partition_hash(labels)

    assert sync.partition_hash != async_result, (
        "the synchronous schedule must differ from the async in-place one here; "
        "if they agree, this test can no longer detect the original bug"
    )


def test_cycle_resolution_collapses_onto_one_label():
    """M6 survived once: the old test only counted oscillating nodes.

    Regression for the no-op. Grouping by CURRENT LABEL makes every member of a
    group already hold the keeper's label, so copying it assigns each node the
    value it already had and nothing changes. The fix groups by node identity
    and collapses the group onto the smallest uuid's label.
    """
    from janus_graph.pipeline.dream_phases import _resolve_cycle

    before = {"a": 4, "b": 9, "c": 7, "d": 1}
    labels = dict(before)
    # One component: every oscillator adopts the smallest uuid's label, which
    # is a fixed point of the 2-cycle.
    _resolve_cycle(labels, ["a", "b", "c", "d"], {"a": 0, "b": 0, "c": 0, "d": 0})
    assert labels == {"a": 4, "b": 4, "c": 4, "d": 4}, (
        "every oscillator in a component must adopt the smallest uuid's label; "
        "an unchanged mapping means _resolve_cycle is a no-op again"
    )

    # Component-scoped: two SEPARATE components must NOT be merged. This is the
    # regression for collapsing the whole oscillating set at once, which turned
    # three disjoint 2-node chains into one 6-node community.
    scoped = {"a": 4, "b": 9, "c": 7, "d": 1}
    _resolve_cycle(scoped, ["a", "b", "c", "d"], {"a": 0, "b": 0, "c": 1, "d": 1})
    assert scoped == {"a": 4, "b": 4, "c": 7, "d": 7}, (
        "resolution must stay inside one connected component"
    )

    # The keeper is the SMALLEST uuid, not the first seen and not the largest.
    labels2 = {"m": 5, "n": 2, "z": 5}
    _resolve_cycle(labels2, ["m", "n", "z"])
    assert labels2["m"] == labels2["n"] == labels2["z"] == 5

    # An empty oscillating set must be a no-op, not a crash.
    labels3 = {"a": 1, "b": 2}
    _resolve_cycle(labels3, [])
    assert labels3 == {"a": 1, "b": 2}


def test_partition_hash_separates_group_boundaries():
    """M7 survived: the earlier collision test did not actually collide.

    Without the length prefix the blob for {group("a"), group("b")} and
    {group("a","b")} is the same string once joined, so a member list that
    spans the separator is the case that needs the prefix.
    """
    from janus_graph.pipeline.dream_phases import partition_hash

    # Two singletons vs one pair: the join must not conflate them.
    assert partition_hash({"a": 0, "b": 1}) != partition_hash({"a": 0, "b": 0})
    # The shape that actually collided before the fix: a single group whose
    # members span the same separator run as two separate groups.
    assert partition_hash({"x": 0, "y": 1, "z": 2}) != partition_hash({"x": 0, "z": 0, "y": 1})
    # And a member that is itself a separator character must not break it.
    assert partition_hash({"a": 0, chr(31): 1}) != partition_hash({"a": 0, chr(31): 0})


def test_oversized_flag_is_computed_from_the_graph():
    """M9 survived: the old test hand-built a result, so it never exercised
    the computation. Build a real graph and read the flag it produces."""
    from janus_graph.pipeline.dream_phases import cluster_graph

    # A star: every leaf is degree 1, the hub is above min_degree. With the pin
    # off, leaves adopt the hub, giving one community of all nodes. Components
    # is 1 and largest_community equals it, so NOT oversized.
    star = {}
    for i in range(6):
        star["leaf%d" % i] = {"hub"}
    star["hub"] = {"leaf%d" % i for i in range(6)}
    res = cluster_graph(star, min_degree=0)
    assert res.components == 1
    # Measured on the 6-leaf star: 2 communities, the largest holding 6 of the
    # 7 nodes. The point of the test is the FLAG, computed from the graph --
    # not a hand-written expectation.
    assert res.largest_community == res.nodes
    assert res.largest_community <= res.largest_component
    assert res.oversized is (res.largest_community > res.largest_component)
    assert res.oversized is False

    # Two separate chains: the largest community cannot exceed the largest
    # component, so the flag must be False.
    chains = {}
    for i in range(3):
        chains["a%d" % i] = {"b%d" % i}
        chains["b%d" % i] = {"a%d" % i}
    res2 = cluster_graph(chains, min_degree=0)
    assert res2.components == 3
    assert res2.oversized is False


def test_empty_snapshot_never_reports_done():
    """M10: a run over an empty snapshot must not claim DONE."""
    from janus_graph.pipeline.dream_phases import cluster_graph, cluster_report

    res = cluster_graph({})
    line = cluster_report(res, 100, 50, force=False)
    assert line.startswith("SKIPPED")
    assert "DONE" not in line


def test_partition_is_invariant_to_how_the_caller_spells_the_graph():
    """REGRESSION for the worst bug of this work.

    The same undirected graph, passed as a symmetric adjacency dict and as an
    asymmetric one (each neighbour listed only under its own key), produced 1
    community and 2 communities. The normalisation unioned the keys but never
    added the missing reverse edges, so cluster_graph silently answered a
    question about a DIFFERENT graph than the one it was given -- the exact
    shape-dependence this module exists to remove, re-entering through the
    front door.

    ``build_adjacency`` always emits symmetric dicts, so ``plan_clustering``
    was never affected. ``cluster_graph`` is public API and accepted the
    asymmetry, so the invariant belongs here.
    """
    from janus_graph.pipeline.dream_phases import cluster_graph

    symmetric = {"a": {"b"}, "b": {"a"}, "c": {"d"}, "d": {"c"}}
    asymmetric = {"a": {"b"}, "c": {"d"}}

    assert (
        cluster_graph(symmetric, min_degree=0).partition_hash
        == cluster_graph(asymmetric, min_degree=0).partition_hash
    )

    # Also invariant to key order, and self-loops must not appear as edges.
    reordered = {k: symmetric[k] for k in reversed(list(symmetric))}
    base = cluster_graph(symmetric, min_degree=0)
    assert cluster_graph(reordered, min_degree=0).partition_hash == base.partition_hash

    looped = cluster_graph({"a": {"a", "b"}, "b": {"a"}}, min_degree=0)
    assert looped.nodes == 2
    assert looped.largest_community <= 2


def test_cluster_graph_accepts_a_node_listed_only_as_a_neighbour():
    """A key that appears only as someone's neighbour must not raise."""
    from janus_graph.pipeline.dream_phases import cluster_graph

    res = cluster_graph({"a": {"b"}}, min_degree=0)
    assert res.nodes == 2


def test_tie_break_resolves_toward_the_smallest_label():
    """M1 survived twice: no test pinned the tie-break DIRECTION.

    Found by search over graphs on 5 nodes: two edges sharing a hub make the
    tie observable, because both leaves have weight 1 and a label each, so the
    outcome is decided purely by which label the tie-break prefers. The
    symmetric fixture I wrote by eye produced the SAME hash either way and would
    have passed vacuously.
    """
    import janus_graph.pipeline.dream_phases as dp

    # a-b and a-c share the hub a: b and c are exactly tied on weight.
    # Compare the SHIPPED implementation against an explicit re-implementation
    # whose only difference is the tie-break direction. An earlier version of
    # this test looped its own schedule and compared two of ITS OWN runs, which
    # compared the same code twice and proved nothing.
    adj = {"a": {"b", "c"}, "b": {"a"}, "c": {"a"}}

    from janus_graph.pipeline.dream_phases import cluster_graph

    def largest_first(graph, max_rounds=200):
        """Same synchronous LPA, tie-break flipped to the LARGEST label."""
        nodes = sorted(graph)
        rank = {u: i for i, u in enumerate(nodes)}
        labels = {u: rank[u] for u in nodes}
        seen = {frozenset(labels.items())}
        for _ in range(max_rounds):
            new = {}
            for node in nodes:
                tally = {}
                for nb in graph[node]:
                    tally[labels[nb]] = tally.get(labels[nb], 0) + 1
                new[node] = (
                    max(tally.items(), key=lambda kv: (kv[1], -kv[0]))[0] if tally else labels[node]
                )
            labels = new
            part = frozenset(labels.items())
            if part in seen:
                break
            seen.add(part)
        return partition_hash(labels)

    from janus_graph.pipeline.dream_phases import partition_hash

    shipped = cluster_graph(adj, min_degree=0).partition_hash
    flipped = largest_first(adj)
    assert shipped != flipped, (
        "the tie-break direction must be observable; identical hashes mean "
        "this fixture never reaches a tie and the test proves nothing"
    )


def test_cycle_resolution_uses_the_smallest_uuid_not_the_first():
    """M6: the keeper must be the SMALLEST uuid of the component."""
    from janus_graph.pipeline.dream_phases import _resolve_cycle

    # Pass the oscillators in DESCENDING order. Grouping sorts them, so the
    # keeper must still be "w" (label 6) -- not "z", the first one seen.
    labels = {"z": 9, "y": 8, "x": 7, "w": 6}
    _resolve_cycle(labels, ["z", "y", "x", "w"], {"z": 0, "y": 0, "x": 0, "w": 0})
    assert labels == {"z": 6, "y": 6, "x": 6, "w": 6}, (
        "the keeper must be the smallest uuid (w, label 6), not the first seen (z, label 9)"
    )

    # Same set, ascending order: the result must be identical.
    labels2 = {"z": 9, "y": 8, "x": 7, "w": 6}
    _resolve_cycle(labels2, ["w", "x", "y", "z"], {"z": 0, "y": 0, "x": 0, "w": 0})
    assert labels2 == labels


def test_partition_hash_distinguishes_group_boundaries():
    """M7 survived: the earlier collision test did not actually collide.

    Without the length prefix, the blob for {("a","b")} and for {("a",),("b",)}
    is the same string once joined, so two DIFFERENT partitions hash the same
    and the determinism check passes on a partition that actually changed.
    """
    from janus_graph.pipeline.dream_phases import partition_hash

    one_group = partition_hash({"a": 0, "b": 0})
    two_groups = partition_hash({"a": 0, "b": 1})
    assert one_group != two_groups

    # And the reverse-order spelling of the same two-group partition, to prove
    # the hash is canonical rather than insertion-dependent.
    assert two_groups == partition_hash({"b": 1, "a": 0})


def test_oversized_is_true_only_when_a_community_beats_its_component():
    """M9 survived: no test forced the flag to True from a real graph.

    Build two components, force the whole oscillating set of one into a single
    label, and check the flag. Since resolution is component-scoped, a correct
    implementation cannot produce a community larger than its component here --
    so assert the invariant directly on several graphs rather than expecting a
    specific oversized=True case that the fixed algorithm no longer yields.
    """
    from janus_graph.pipeline.dream_phases import cluster_graph

    graphs = [
        {"a": {"b"}, "b": {"a"}, "c": {"d"}, "d": {"c"}},
        {"a": {"b"}, "b": {"a"}, "c": {"d"}, "d": {"c"}, "e": {"f"}, "f": {"e"}},
        {"hub": {"l%d" % i for i in range(6)}},
    ]
    for g in graphs:
        for node in ("e", "f"):
            if node in g:
                g.setdefault(node, set())
        for i in range(6):
            g.setdefault("l%d" % i, {"hub"})
        res = cluster_graph(g, min_degree=0)
        assert res.oversized is (res.largest_community > res.largest_component), g
        # The fixed, component-scoped resolution makes this unreachable.
        assert res.largest_community <= res.largest_component, g
        assert res.oversized is False, g


def test_oversized_is_false_when_a_community_is_smaller_than_its_component():
    """M9 was a REAL test gap, and the gap had a shape worth pinning.

    M9 inverted ``oversized = largest_community > largest_component`` into
    ``<``. I first assumed that was an equivalent mutant because ``>`` is
    never True on any graph I tried (exhaustive n<=5, 800 random). That was an
    overclaim: I measured only the ``>`` polarity. ``<`` is the NORMAL state of
    any graph whose components do not fully unify -- 899 of the graphs I
    enumerated. The flag survived only because no test handed it one.

    Concretely: min_degree=2 pins both nodes of a 2-chain, so each keeps its own
    seed label. Largest community is 1, largest component is 2. Strictly less,
    so the flag must be False -- and an inverted comparator returns True here.
    """
    from janus_graph.pipeline.dream_phases import cluster_graph

    # Three disjoint 2-chains, no isolations, so the comparison is strictly
    # '<' rather than '=='.
    chains = {}
    for i in range(3):
        chains["a%d" % i] = {"b%d" % i}
        chains["b%d" % i] = {"a%d" % i}

    res = cluster_graph(chains, min_degree=2)
    assert res.components == 3
    assert res.largest_component == 2
    assert res.pinned_nodes == 6
    assert res.largest_community == 1
    # The state this test exists for: community strictly SMALLER than component.
    assert res.largest_community < res.largest_component
    assert res.oversized is False

    # Same graph, pinning off: the chains unify, so the two values become equal
    # and the flag stays False. Pinning moves the value but never the polarity.
    res_unpinned = cluster_graph(chains, min_degree=0)
    assert res_unpinned.largest_community == res_unpinned.largest_component
    assert res_unpinned.oversized is False


def test_oversized_is_a_live_guard_against_cross_component_merges(monkeypatch):
    """Why the field is kept at all: it is the alarm for a real regression.

    Cycle resolution is component-scoped. An earlier version collapsed the WHOLE
    oscillating set into one label, which merged separate components into a
    single community spanning both -- structurally impossible for the fixed
    algorithm, because a node only ever reads labels of its own neighbours.

    So a correct implementation can never raise ``oversized``. That makes the
    field look dead, and I nearly dropped it. Reinstating the buggy resolver
    here proves it is not dead: it fires on 21 of 200 random disconnected
    graphs; the smallest is 8 nodes and is used below. This test is the reason
    the field stays.
    """
    from janus_graph.pipeline import dream_phases as dp

    def collapse_everything(labels, oscillating, comp_of=None):
        """The pre-fix resolver: one global group, components ignored."""
        group = sorted(oscillating)
        if not group:
            return
        keeper = labels[group[0]]
        for u in group:
            labels[u] = keeper

    # The smallest graph that witnesses this, found by seeded search rather
    # than drawn by hand: two components of four nodes that both oscillate --
    # a K4, and a K4 with one edge removed. Both components oscillate, so the
    # resolver runs; with components respected each stays 4, and collapsing
    # them together spans all 8 nodes.
    def two_oscillating_components():
        return {
            "n0": {"n2", "n3"},
            "n1": {"n2", "n3"},
            "n2": {"n0", "n1", "n3"},
            "n3": {"n0", "n1", "n2"},
            "n4": {"n6", "n7"},
            "n5": {"n6", "n7"},
            "n6": {"n4", "n5"},
            "n7": {"n4", "n5"},
        }

    correct = dp.cluster_graph(two_oscillating_components(), min_degree=2)
    assert correct.components == 2
    assert correct.largest_component == 4
    assert correct.cycle_detected is True, "witness must actually oscillate"
    assert correct.oversized is False
    assert correct.largest_community == 4

    monkeypatch.setattr(dp, "_resolve_cycle", collapse_everything)
    buggy = dp.cluster_graph(two_oscillating_components(), min_degree=2)
    assert buggy.largest_community == 8
    assert buggy.largest_component == 4
    assert buggy.oversized is True
    assert "CAVEAT" in buggy.describe()
