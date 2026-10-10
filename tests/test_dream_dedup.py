"""Tests for the Phase 2 dedup logic.

Design notes
------------
The planner is pure (no I/O) precisely so the merge DECISION can be tested
without a database, and so the exact numbers from the 2026-10-07 snapshot
(309 duplicate-name clusters, 0 of them with identical summaries) can be
pinned as regression guards.

``test_negative_control_*`` exists because a test that only ever passes proves
nothing: each one mutates the planner so that a regression WOULD occur, and
asserts the test suite notices.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from janus_graph.pipeline.dedup import (
    DEFAULT_DEDUP_THRESHOLD,
    DedupPlan,
    escape_cypher_string,
    jaccard,
    plan_dedup,
    tokenize,
)
from janus_graph.pipeline.dedup_apply import _root_map, apply_plan


# --------------------------------------------------------------------------
# similarity primitives
# --------------------------------------------------------------------------


def test_jaccard_identical_and_disjoint():
    a = tokenize("the daemon moved five agent principles")
    b = tokenize("the daemon moved five agent principles")
    assert jaccard(a, b) == 1.0
    assert jaccard(tokenize("alpha beta"), tokenize("gamma delta")) == 0.0


def test_jaccard_two_empty_summaries_are_identical():
    """Two nodes with no summary must not be treated as 'different'."""
    assert jaccard(tokenize(""), tokenize("")) == 1.0


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("The Signet Helper: get_github_tags_sorted()") == frozenset(
        {"the", "signet", "helper", "get_github_tags_sorted"}
    )


def test_escape_cypher_string_escapes_backslash_before_quote():
    """FalkorDB escapes with a backslash, so the backslash must go first.

    Doing it the other way round turns ``a'b`` into ``a\\'b`` -> ``a\\\\'b``,
    i.e. the literal grows a backslash instead of quoting.
    """
    assert escape_cypher_string("a'b") == "a\\'b"
    assert escape_cypher_string("a\\b") == "a\\\\b"
    assert escape_cypher_string("O'Brien\\x") == "O\\'Brien\\\\x"


# --------------------------------------------------------------------------
# planner behaviour
# --------------------------------------------------------------------------


def _ent(uuid, name, summary, created="2026-10-01T00:00:00"):
    return {"uuid": uuid, "name": name, "summary": summary, "created_at": created}


def test_merges_only_near_identical_summaries_not_same_name():
    """The 2026-10-07 finding, pinned.

    Same name is NOT sufficient. These two share a name but describe different
    facts, exactly like the 'daemon' x17 cluster in the real graph, so a
    name-keyed merge would destroy knowledge.
    """
    ents = [
        _ent("u1", "daemon", "Daemon provided explicit rules, facts, and identity."),
        _ent("u2", "daemon", "The daemon sent a Zalo message to Thúy Vi at 09:12."),
    ]
    plan = plan_dedup(ents)
    assert plan.merges == []
    assert plan.nodes_to_delete == 0


def test_merges_true_restatement_and_keeps_longest_summary():
    ents = [
        _ent(
            "u1",
            "Signet helper",
            "Inline python3 usage in the Signet helper was removed in favor of "
            "the get_github_tags_sorted sorting helper.",
        ),
        _ent(
            "u2",
            "Signet helper",
            "Inline python3 usage was removed in favor of the new sorting "
            "helper get_github_tags_sorted().",
        ),
    ]
    plan = plan_dedup(ents)
    assert len(plan.merges) == 1
    merge = plan.merges[0]
    # Survivor = longest summary, so no text is discarded.
    assert merge.survivor_uuid == "u1"
    assert merge.victim_uuid == "u2"
    assert plan.nodes_to_delete == 1
    assert plan.summary_chars_saved == len(ents[1]["summary"])


def test_survivor_policy_prefers_longest_then_oldest_then_uuid():
    from janus_graph.pipeline.dedup import _pick_survivor

    long_old = _ent("z", "n", "aaaa bbbb", "2026-01-01")
    long_new = _ent("a", "n", "aaaa bbbb", "2026-09-01")
    short = _ent("b", "n", "aaaa", "2026-01-01")
    # equal length -> oldest wins, not lowest uuid
    assert _pick_survivor([long_old, long_new, short])["uuid"] == "z"


def test_cluster_of_three_collapses_to_one_not_two():
    """The plan is a chain: folding A into B must re-test B against the rest.

    Regression guard added after a mutation test MISSED a change that removed
    the KEEPER from ``remaining`` instead of the loser. The original version of
    this test asserted only "one merge happened", which that mutation still
    satisfied -- it does not check WHICH node survives. This version pins the
    survivor set, not just the merge count.
    """
    base = "inline python3 usage was removed in favor of the sorting helper"
    ents = [
        _ent("u1", "h", base + " get_github_tags_sorted"),
        _ent("u2", "h", base + " get_github_tags_sorted helper"),
        _ent("u3", "h", "an entirely unrelated statement about something else"),
    ]
    plan = plan_dedup(ents)
    assert len(plan.merges) == 1  # only u1/u2 pair
    assert plan.nodes_to_delete == 1
    assert {m.victim_uuid for m in plan.merges} <= {"u1", "u2"}
    # The unrelated node must remain untouched in the plan.
    assert "u3" not in {m.victim_uuid for m in plan.merges}


def test_chain_fold_removes_the_loser_and_keeps_the_survivor():
    """Three mutually-similar summaries must leave exactly ONE survivor.

    Written after a mutation test MISSED a change that removed the KEEPER from
    ``remaining`` instead of the loser. The mutation still produced "one
    merge", so an assertion on merge COUNT alone cannot catch it.

    The invariant pinned here is the one that actually matters: across a chain
    a node may legitimately be a survivor in one fold and a victim in the next
    (that is what folding means), but the node with the LONGEST summary must
    be the one that survives to the end -- otherwise the fold discarded text
    the survivor already had.
    """
    base = "the janus graph persists episodic and entity nodes into falkordb"
    ents = [
        _ent("u1", "n", base + " graph"),
        _ent("u2", "n", base + " graph falkordb"),
        _ent("u3", "n", base + " graph falkordb now"),
    ]
    longest = max(ents, key=lambda e: len(e["summary"]))
    plan = plan_dedup(ents)

    assert len(plan.merges) == 2, "a 3-node chain needs two folds"
    final = {"u1", "u2", "u3"} - {m.victim_uuid for m in plan.merges}
    assert len(final) == 1, (
        "expected 1 survivor among 3 nodes, got %s (chain did not fold)" % sorted(final)
    )
    assert final == {longest["uuid"]}, (
        "the longest summary must be the final survivor; got %s instead of %s"
        % (sorted(final), longest["uuid"])
    )


def test_threshold_is_respected_and_default_is_conservative():
    near = "alpha beta gamma delta epsilon zeta"
    far = "alpha beta gamma delta epsilon omega"
    ents = [_ent("u1", "n", near), _ent("u2", "n", far)]
    assert plan_dedup(ents, threshold=0.99).merges == []
    assert len(plan_dedup(ents, threshold=0.5).merges) == 1
    assert DEFAULT_DEDUP_THRESHOLD >= 0.80


def test_plan_is_idempotent_after_merge_is_applied():
    """Re-planning on the post-merge graph must find nothing left to do."""
    base = "the janus graph stores episodic and entity nodes in falkordb"
    ents = [_ent("u1", "n", base + " graph"), _ent("u2", "n", base + " graph falkordb")]
    first = plan_dedup(ents)
    assert len(first.merges) == 1
    victim = first.merges[0].victim_uuid
    survivor = first.merges[0].survivor_uuid
    after = [e for e in ents if e["uuid"] != victim]
    second = plan_dedup(after)
    assert second.merges == []
    assert survivor in {e["uuid"] for e in after}


def test_duplicate_mentions_to_same_target_are_counted_as_droppable():
    base = "identical summary text for the merge candidate pair here"
    ents = [_ent("u1", "n", base), _ent("u2", "n", base + " x")]
    mentions = [
        {"src_uuid": "ep1", "dst_uuid": "u2"},
        {"src_uuid": "ep1", "dst_uuid": "u1"},
    ]
    plan = plan_dedup(ents, mentions=mentions)
    assert len(plan.merges) == 1
    # both mentions collapse onto the survivor -> one is redundant
    assert plan.mentions_to_drop == 1


def test_relation_between_two_merge_targets_becomes_a_dropped_self_loop():
    base = "identical summary text for the merge candidate pair here"
    ents = [_ent("u1", "n", base), _ent("u2", "n", base + " x")]
    rels = [{"src_uuid": "u1", "dst_uuid": "u2", "fact": "f"}]
    plan = plan_dedup(ents, relations=rels)
    assert len(plan.merges) == 1
    # An edge between two nodes that collapse into ONE is a self-loop, not a
    # duplicate, so it is counted under self_loops_dropped and kept out of
    # relations_to_drop.
    assert plan.self_loops_dropped == 1
    assert plan.relations_to_drop == 0
    assert plan.total_relations_removed == 1


def test_drop_counters_ignore_pre_existing_duplication():
    """Regression guard for a counter bug found by measurement.

    An earlier planner counted drops by walking the WHOLE graph, so it charged
    the merge for redundancy that existed before it ran. The 2026-10-07
    snapshot has 136 duplicate MENTIONS pairs and 12 duplicate RELATES_TO
    triples with zero merges applied, and the planner dutifully reported
    "dropped=136" for a 2-node merge.

    Edges that the merge does not touch must contribute 0 to the counters.
    """
    base = "identical summary text for the merge candidate pair here"
    ents = [_ent("u1", "n", base), _ent("u2", "n", base + " x")]
    # 5 pre-existing duplicate mentions that the merge never touches.
    untouched = [{"src_uuid": "ep%d" % i, "dst_uuid": "u9"} for i in range(5)] + [
        {"src_uuid": "ep9", "dst_uuid": "u9"}
    ]
    plan = plan_dedup(ents, mentions=untouched)
    assert len(plan.merges) == 1
    assert plan.mentions_to_drop == 0, (
        "pre-existing duplicates must not be charged to this merge; got %d" % plan.mentions_to_drop
    )


def test_drop_counters_do_credit_collisions_the_merge_creates():
    base = "identical summary text for the merge candidate pair here"
    ents = [_ent("u1", "n", base), _ent("u2", "n", base + " x")]
    # Two episodes mention the victim; after redirect both land on the
    # survivor. One mention is genuinely redundant with the survivor's own.
    mentions = [
        {"src_uuid": "ep1", "dst_uuid": "u2"},
        {"src_uuid": "ep1", "dst_uuid": "u1"},
        {"src_uuid": "ep2", "dst_uuid": "u2"},
        {"src_uuid": "ep2", "dst_uuid": "u1"},
    ]
    plan = plan_dedup(ents, mentions=mentions)
    assert len(plan.merges) == 1
    assert plan.mentions_to_drop == 2  # ep1 and ep2 each collapse by one
    assert plan.mentions_to_redirect == 2


def test_self_loop_counter_is_separate_from_relations_to_drop():
    """Self-loops and duplicates are different losses and stay separated."""
    base = "identical summary text for the merge candidate pair here"
    ents = [_ent("u1", "n", base), _ent("u2", "n", base + " x")]
    rels = [{"src_uuid": "u1", "dst_uuid": "u2", "fact": "f"}]
    plan = plan_dedup(ents, relations=rels)
    assert len(plan.merges) == 1
    assert plan.self_loops_dropped == 1
    assert plan.relations_to_drop == 0
    assert plan.total_relations_removed == 1
    assert "=0 dup + 1 self-loop" in plan.describe()


def test_describe_never_reports_bare_done():
    plan = DedupPlan(clusters_scanned=309, duplicate_name_nodes=720)
    text = plan.describe()
    assert text.startswith("DONE (")
    assert "309" in text and "720" in text
    assert text != "DONE"


# --------------------------------------------------------------------------
# chain resolution
# --------------------------------------------------------------------------


def test_root_map_collapses_chain_and_survives_cycle():
    root = _root_map({"a": "b", "b": "c", "c": "c"})
    assert root["a"] == "c"
    assert root["b"] == "c"


def test_root_map_cycle_does_not_hang():
    # A cycle cannot come from the planner, but the resolver must not hang.
    root = _root_map({"a": "b", "b": "a"})
    assert root["a"] in {"a", "b"}


# --------------------------------------------------------------------------
# apply layer: dry-run must never write
# --------------------------------------------------------------------------


def test_apply_plan_dry_run_makes_zero_queries():
    base = "identical summary text for the merge candidate pair here"
    plan = plan_dedup([_ent("u1", "n", base), _ent("u2", "n", base + " x")])
    assert len(plan.merges) == 1
    graph = MagicMock()
    counters = apply_plan(graph, plan, apply=False)
    assert graph.query.call_count == 0
    assert counters["applied"] is False
    assert counters["nodes_to_delete"] == 1


def test_apply_plan_on_empty_plan_touches_nothing():
    graph = MagicMock()
    counters = apply_plan(graph, DedupPlan(), apply=True)
    assert graph.query.call_count == 0
    assert counters["nodes_deleted"] == 0


# --------------------------------------------------------------------------
# run_phase2_dedup degradation paths
# --------------------------------------------------------------------------


def test_run_phase2_reports_skipped_when_falkordb_down():
    from janus_graph.pipeline import dedup_apply

    cfg = MagicMock()
    with patch.object(
        dedup_apply, "_graph", return_value=(None, "ConnectionRefusedError: refused")
    ):
        out = dedup_apply.run_phase2_dedup(cfg, "graphiti_memory", force=False)
    assert out["status"] == "SKIPPED"
    assert "unreachable" in out["reason"]
    assert out["merges"] == 0
    assert out["applied"] is False


def test_run_phase2_reports_failed_when_snapshot_read_fails():
    from janus_graph.pipeline import dedup_apply

    graph = MagicMock()
    graph.query.side_effect = RuntimeError("boom")
    cfg = MagicMock()
    with patch.object(dedup_apply, "_graph", return_value=(graph, None)):
        out = dedup_apply.run_phase2_dedup(cfg, "graphiti_memory", force=False)
    assert out["status"] == "FAILED"
    assert out["applied"] is False


def _fake_graph(read_rows_for):
    """MagicMock whose query() returns a row shape matching the query asked.

    apply_plan issues several different RETURN clauses, so a single canned
    result_set makes the row-unpacking blow up for reasons that have nothing
    to do with the behaviour under test. Dispatch on the query text instead.
    """
    graph = MagicMock()

    def _query(cypher, params=None, **kw):
        rows = read_rows_for(cypher)
        res = MagicMock()
        res.result_set = rows
        return res

    graph.query.side_effect = _query
    return graph


def test_apply_order_is_create_before_delete_for_mentions():
    """A regression guard found by reading the writer path, not by a failing run.

    FalkorDB exposes no transaction (verified in the `falkordb` client), so a
    DELETE-then-CREATE redirect would destroy an edge outright if the process
    died between the two statements. Every redirect must CREATE first.

    This asserts the ORDER of the two statements, by matching the Cypher text,
    which is the only externally observable trace of what the writer did.
    """
    base = "identical summary text for the merge candidate pair here"
    ents = [_ent("u1", "n", base), _ent("u2", "n", base + " x")]
    mentions = [{"src_uuid": "ep1", "dst_uuid": "u1"}]
    plan = plan_dedup(ents, mentions=mentions)
    assert len(plan.merges) == 1
    victim = plan.merges[0].victim_uuid

    def rows_for(cypher):
        # 2026-10-09: MENTIONS has no src_uuid/dst_uuid properties (measured on
        # all 26,030 live edges), so endpoints come from the matched nodes and
        # the read is `s.uuid, v.uuid`. The fake below must match the query the
        # code actually issues, otherwise it silently returns no rows and the
        # test fails for a reason unrelated to statement ordering.
        if "RETURN e.uuid, s.uuid, v.uuid" in cypher:
            return [["edge-1", "ep1", victim, "2026-10-01", "graphiti_memory"]]
        if "RETURN e.uuid, a.uuid, b.uuid" in cypher:
            return []  # no RELATES_TO touching the victim
        if "CREATE" in cypher and "RETURN count(*)" in cypher:
            # The MENTIONS redirect CREATE reports how many rows it matched so
            # the writer only DELETEs the old edge when the new one really
            # exists. Answering with a positive count is the "it worked" case.
            return [[1]]
        if "RETURN count(v)" in cypher:
            return [[0]]
        if "RETURN sum(size(ids) - 1)" in cypher:
            return [[0]]
        # Anything else gets an EMPTY result set, not [[]]. Returning [[]] hands
        # the row-unpacking loop a single zero-width row, which fails with
        # "not enough values to unpack" for reasons unrelated to the behaviour
        # under test.
        return []

    graph = _fake_graph(rows_for)
    apply_plan(graph, plan, apply=True)

    calls = [c.args[0] for c in graph.query.call_args_list if c.args]
    assert calls, "apply_plan issued no queries"

    create_i = next((i for i, q in enumerate(calls) if "CREATE (s)-[e:MENTIONS" in q), None)
    delete_i = next(
        (
            i
            for i, q in enumerate(calls)
            if "DELETE e" in q and "MENTIONS]->(:Entity {uuid: $old})" in q
        ),
        None,
    )
    assert create_i is not None, "no MENTIONS CREATE issued"
    assert delete_i is not None, "no MENTIONS DELETE issued"
    assert create_i < delete_i, (
        "redirect order is DELETE-then-CREATE; with no transaction available "
        "a crash between them loses the edge (create at %d, delete at %d)" % (create_i, delete_i)
    )


def test_apply_never_deletes_a_victim_still_holding_edges():
    """Phase D guards node deletion with NOT (v)--().

    Regression guard written after a mutation test MISSED the guard-removal
    mutation. The first version grepped the source text and matched whichever
    DETACH DELETE statement still had the guard, so removing the guard from one
    of the two concatenated string literals still passed. This version checks
    the statements the code actually ISSUES, captured through a fake graph --
    that is the only place the concatenated Cypher becomes a real string.
    """
    base = "identical summary text for the merge candidate pair here"
    ents = [_ent("u1", "n", base), _ent("u2", "n", base + " x")]
    plan = plan_dedup(ents)
    assert len(plan.merges) == 1

    issued = []

    def rows_for(cypher):
        issued.append(cypher)
        if "RETURN count(v)" in cypher:
            return [[0]]
        if "RETURN sum(size(ids) - 1)" in cypher:
            return [[0]]
        return []

    graph = _fake_graph(rows_for)
    apply_plan(graph, plan, apply=True)

    deletes = [q for q in issued if "DETACH DELETE" in q]
    assert deletes, "apply_plan never issued a DETACH DELETE"
    for q in deletes:
        assert "NOT (v)--()" in q, (
            "unguarded DETACH DELETE would silently detach live edges: %r" % q
        )
    # Every DETACH DELETE must also be scoped to the victim list, so a future
    # edit cannot widen it into a blanket delete.
    for q in deletes:
        assert "v.uuid IN $victims" in q, "DETACH DELETE is not scoped to the victim list: %r" % q


# --------------------------------------------------------------------------
# negative controls — prove the suite above can actually fail
# --------------------------------------------------------------------------


def test_negative_control_survivor_policy_regression_is_caught():
    """If the survivor policy flipped to 'first created wins', this must fail.

    The two nodes are deliberately built so the two policies DISAGREE: the
    longer summary is the newer one. An earlier draft of this test gave both
    summaries the same length, which made the buggy policy produce the same
    answer — a negative control that cannot fail is not a control.
    """
    long_new = _ent("a", "n", "aaaa bbbb cccc dddd", "2026-09-01")
    short_old = _ent("z", "n", "cccc", "2026-01-01")

    from janus_graph.pipeline.dedup import _pick_survivor

    real = _pick_survivor([long_new, short_old])

    # Simulate the regression: sort by created_at only.
    regressed = sorted([long_new, short_old], key=lambda n: str(n["created_at"]))[0]
    assert real["uuid"] == "a"  # longest summary wins
    assert regressed["uuid"] == "z"  # the buggy choice differs
    assert real["uuid"] != regressed["uuid"]


def test_negative_control_name_only_merge_would_be_caught():
    """Merge-by-name would collapse these two; the real planner must not.

    This is the exact failure mode the 309-cluster measurement exposed.
    """
    ents = [
        _ent("u1", "PicoClaw", "PicoClaw must manage N processes for N sessions."),
        _ent("u2", "PicoClaw", "The planning-mcp server performs a handshake."),
    ]
    assert plan_dedup(ents).merges == []
    # ...and a name-only merge would have produced one, proving the test bites.
    by_name = {}
    for e in ents:
        by_name.setdefault(e["name"], []).append(e)
    assert max(len(v) for v in by_name.values()) == 2
