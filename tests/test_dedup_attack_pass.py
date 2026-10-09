"""Attack pass: try to break the P2 fix before anh sees it.

Attack 1 (DATA LOSS): Phase A issues CREATE then DELETE with no transaction.
If the CREATE silently matches nothing, the DELETE still runs and the MENTIONS
is gone. Before the fix this was harmless because the CREATE never matched
anything either -- now it CAN match, so the ordering guard is load-bearing for
the first time. Pin: only delete after a CREATE that actually created a row.

Attack 2 (PHANTOM WORK): the counter must not report a move that did not happen.

Attack 3 (UNRESOLVED ENDPOINT): if an endpoint cannot be resolved, the edge must
be SKIPPED with an error, never moved against a NULL uuid.

Attack 4 (NEW INVENTED PROPERTIES): the CREATE must not reintroduce
src_uuid/dst_uuid on MENTIONS -- that is the bug itself.
"""
import re

from janus_graph.pipeline.dedup import plan_dedup
from janus_graph.pipeline.dedup_apply import apply_plan, load_graph_snapshot
from tests.test_dedup_mentions_topology import (
    EPISODIC_UUID,
    SUMMARY_A,
    SUMMARY_B,
    SURVIVOR_UUID,
    VICTIM_UUID,
    _FakeFalkor,
)

_MENTION_PROPS = {"uuid", "group_id", "created_at"}


def _plan():
    return plan_dedup(
        [
            {"uuid": SURVIVOR_UUID, "name": "CRG_TOOLS", "summary": SUMMARY_A},
            {"uuid": VICTIM_UUID, "name": "CRG_TOOLS", "summary": SUMMARY_B},
        ],
        [{"uuid": "e1", "src_uuid": EPISODIC_UUID, "dst_uuid": VICTIM_UUID,
          "created_at": None, "group_id": "gid"}],
        [],
    )


class _CreateMatchesNothing(_FakeFalkor):
    """CREATE binds a real uuid but the MATCH still yields 0 rows."""

    def query(self, cypher, params=None):
        if "CREATE" in cypher:
            self.creates.append((cypher, params or {}))
            return type("R", (), {"result_set": [[0]]})()
        return super().query(cypher, params)


def _mention_creates(g):
    return [(q, p) for q, p in g.creates if "MENTIONS" in q]


def _mention_deletes(g):
    """MENTIONS edge deletions only -- Phase D DETACH DELETE is a node delete."""
    return [(q, p) for q, p in g.deletes if "MENTIONS" in q]


class _CreateRaises(_FakeFalkor):
    def query(self, cypher, params=None):
        if "CREATE" in cypher:
            raise RuntimeError("disk full (simulated)")
        return super().query(cypher, params)


def test_attack1_no_delete_when_create_matched_nothing():
    """If CREATE creates 0 rows, the old edge MUST survive."""
    g = _CreateMatchesNothing()
    counters = apply_plan(g, _plan(), "gid")
    assert counters["mentions_moved"] == 0
    assert _mention_creates(g), "the CREATE should still have been attempted"
    assert not _mention_deletes(g), (
        "DELETE ran after a CREATE that created nothing: the MENTIONS edge "
        "would be destroyed outright, with no transaction to roll it back"
    )
    assert any("endpoints did not match" in e for e in counters["errors"])


def test_attack1b_no_delete_when_create_raises():
    g = _CreateRaises()
    counters = apply_plan(g, _plan(), "gid")
    assert counters["mentions_moved"] == 0
    assert not _mention_deletes(g)
    assert counters["errors"], "a failed CREATE must be reported, not swallowed"


def test_attack2_counter_matches_reality():
    g = _FakeFalkor()
    counters = apply_plan(g, _plan(), "gid")
    created = len(_mention_creates(g))
    deleted = len(_mention_deletes(g))
    assert counters["mentions_moved"] == created == deleted


def test_attack3_unresolved_endpoint_is_skipped_with_an_error():
    """A row whose endpoints do not resolve must not be moved at all."""

    class _NullEndpoints(_FakeFalkor):
        def query(self, cypher, params=None):
            if "RETURN e.uuid, s.uuid, v.uuid" in cypher:
                return type("R", (), {"result_set": [["e1", None, None, None, "gid"]]})()
            return super().query(cypher, params)

    g = _NullEndpoints()
    counters = apply_plan(g, _plan(), "gid")
    assert counters["mentions_moved"] == 0
    assert not _mention_creates(g) and not _mention_deletes(g)
    assert any("unresolved" in e for e in counters["errors"]), counters["errors"]


def test_attack4_create_does_not_reintroduce_invented_properties():
    """MENTIONS must keep exactly the properties the schema really has."""
    g = _FakeFalkor()
    apply_plan(g, _plan(), "gid")
    assert _mention_creates(g)
    cypher = _mention_creates(g)[0][0]
    written = set(re.findall(r"(\w+):\s*\$", cypher))
    assert "src_uuid" not in written
    assert "dst_uuid" not in written
    assert written <= {"uuid", "group_id", "created_at"}, (
        "CREATE wrote properties MENTIONS does not have: %s" % (written,)
    )


def test_attack5_snapshot_matches_measured_schema():
    """The fake must stay faithful to the measured edge shape."""
    assert _MENTION_PROPS == {"uuid", "group_id", "created_at"}
    snap, errs = load_graph_snapshot(_FakeFalkor(), "gid")
    assert errs == ""
    m = snap["mentions"][0]
    assert m["src_uuid"] == EPISODIC_UUID and m["dst_uuid"] == VICTIM_UUID
