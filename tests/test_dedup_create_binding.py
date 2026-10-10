"""Bug 3: the CREATE attaches the edge to NEW empty nodes, not the matched ones.

The 2026-10-09 fix read MENTIONS endpoints from the topology (``s.uuid``,
``v.uuid``) and guarded the DELETE behind ``RETURN count(*)``. Both are correct
-- and together they shipped a THIRD bug into the live graph:

    MATCH (:Episodic {uuid: $src}), (:Entity {uuid: $new})
    CREATE (s)-[e:MENTIONS {...}]->(t)

The MATCH nodes are *anonymous*. In openCypher a node only becomes a bound
variable if it is named, so ``(s)`` and ``(t)`` in the CREATE clause are not the
matched Episodic/Entity -- they are brand new, empty, unlabelled nodes. The edge
keeps its uuid, so ``RETURN count(*)`` returns 1 and the guard added in the same
commit passes happily.

Measured on the live graph after one apply: 2 MENTIONS edges were attached to 4
empty nodes (labels(a) == labels(b) == []), so
``MATCH (a:Episodic)-[:MENTIONS]->(b:Entity)`` saw 26,084 of 26,086 edges. The
count-preserving checks all passed; only an endpoint-labelled query caught it.

The fake graphs in the other dedup tests cannot catch this: their CREATE branch
returns ``[[1]]`` without modelling node creation. This fake DOES model it --
``_create_nodes`` stands for the labels a freshly created node would carry.
"""

from __future__ import annotations

import re

from janus_graph.pipeline.dedup import plan_dedup
from janus_graph.pipeline.dedup_apply import apply_plan

SUMMARY_A = "CRG_TOOLS exposes build embed and refactor tools for the graph today"
SUMMARY_B = "CRG_TOOLS exposes build embed and refactor tools for the graph"

EPISODIC_UUID = "ep-1"
VICTIM_UUID = "vict"
SURVIVOR_UUID = "surv"
MENTION_EDGE_UUID = "e1"


class _Result:
    def __init__(self, rows):
        self.result_set = rows


def _match_aliases(cypher: str) -> list:
    """Names bound by the MATCH clause, in order of appearance.

    An anonymous node ``(:Episodic {...})`` binds nothing; a named one
    ``(s:Episodic {...})`` binds ``s``. This is the whole bug.
    """
    match_clause = cypher.split("CREATE", 1)[0]
    return re.findall(r"\(\s*([A-Za-z_][A-Za-z0-9_]*)?\s*:", match_clause)


class _CypherAwareFalkor:
    """Fake that models the anonymous-MATCH-node CREATE bug faithfully."""

    def __init__(self):
        self.queries = []
        self.created_edges = []  # (alias_src, alias_dst, uuid, labels)
        self.deleted = []
        self.mentions = [
            (EPISODIC_UUID, VICTIM_UUID, {"uuid": MENTION_EDGE_UUID, "group_id": "gid"})
        ]
        # node uuid -> labels. An empty list == an unlabelled placeholder.
        self.node_labels = {
            EPISODIC_UUID: ["Episodic"],
            VICTIM_UUID: ["Entity"],
            SURVIVOR_UUID: ["Entity"],
        }

    def query(self, cypher, params=None):
        params = params or {}
        self.queries.append(cypher)

        if "RETURN keys(e)" in cypher:
            return _Result([[["uuid", "group_id", "created_at"]]])

        if "CREATE" in cypher:
            bound = _match_aliases(cypher)
            # Whatever the CREATE names -- (s), (t) -- is bound ONLY if the
            # MATCH named it. Otherwise a fresh empty node comes into being.
            create_src = re.search(r"CREATE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)", cypher)
            create_dst = re.search(r"->\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)?", cypher)
            src_alias = create_src.group(1) if create_src else None
            dst_alias = create_dst.group(1) if create_dst else None

            def endpoint(alias):
                if alias and alias in bound:
                    return self._uuid_for(alias, params), list(
                        self.node_labels.get(self._uuid_for(alias, params), [])
                    )
                # unbound: a brand new, empty, unlabelled node
                self._next_placeholder = getattr(self, "_next_placeholder", 9000) + 1
                new_id = "placeholder-%d" % self._next_placeholder
                self.node_labels[new_id] = []
                return new_id, []

            src_id, src_labels = endpoint(src_alias)
            dst_id, dst_labels = endpoint(dst_alias)
            eu = params.get("eu") or params.get("eu")
            self.created_edges.append((src_id, dst_id, eu, src_labels, dst_labels))
            if "RETURN count(*)" in cypher:
                return _Result([[1]])
            return _Result([])

        if "DELETE" in cypher:
            self.deleted.append((cypher, params))
            return _Result([[1]])

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

        if cypher.strip().startswith("MATCH (n:Entity)"):
            return _Result(
                [
                    [k, "CRG_TOOLS", SUMMARY_A if k == SURVIVOR_UUID else SUMMARY_B, None]
                    for k in (SURVIVOR_UUID, VICTIM_UUID)
                ]
            )

        if "count(" in cypher:
            return _Result([[0]])

        return _Result([])

    def _uuid_for(self, alias, params):
        return {
            "s": params.get("src"),
            "t": params.get("new"),
            "a": params.get("s2"),
            "b": params.get("d2"),
        }.get(alias)

    def _resolve(self, item, src, dst, props):
        alias, _, prop = item.strip().partition(".")
        if alias == "e":
            return props.get(prop) if prop in ("uuid", "group_id", "created_at") else None
        if alias in ("s", "a"):
            return src
        if alias in ("n", "v", "b", "t"):
            return dst
        return None


def _plan():
    return plan_dedup(
        [
            {"uuid": SURVIVOR_UUID, "name": "CRG_TOOLS", "summary": SUMMARY_A, "created_at": None},
            {"uuid": VICTIM_UUID, "name": "CRG_TOOLS", "summary": SUMMARY_B, "created_at": None},
        ],
        [
            {
                "uuid": MENTION_EDGE_UUID,
                "src_uuid": EPISODIC_UUID,
                "dst_uuid": VICTIM_UUID,
                "created_at": None,
                "group_id": "gid",
            }
        ],
        [],
    )


def _apply():
    g = _CypherAwareFalkor()
    counters = apply_plan(g, _plan(), apply=True)
    return g, counters


def test_create_binds_the_matched_nodes_not_new_placeholders():
    """The MENTIONS CREATE must attach to the MATCHed Episodic and Entity.

    Regression pin for the 2026-10-09 live incident: the MATCH nodes were
    anonymous, so the edge landed on two empty placeholders and became
    invisible to every labelled query while every count stayed correct.
    """
    g, counters = _apply()
    assert counters["mentions_moved"] == 1

    creates = [c for c in g.created_edges if c[2] == MENTION_EDGE_UUID]
    assert creates, "apply_plan must issue the CREATE half of the move"
    src_id, dst_id, _, src_labels, dst_labels = creates[0]

    assert src_id == EPISODIC_UUID, (
        "CREATE built a NEW node instead of using the matched Episodic: %r" % (src_id,)
    )
    assert dst_id == SURVIVOR_UUID, (
        "CREATE built a NEW node instead of using the matched Entity: %r" % (dst_id,)
    )
    assert src_labels == ["Episodic"], (
        "the MENTIONS source lost its label -> unreachable by "
        "(a:Episodic)-[:MENTIONS]->(b:Entity): %r" % (src_labels,)
    )
    assert dst_labels == ["Entity"], "the MENTIONS target lost its label: %r" % (dst_labels,)


def _move_queries() -> list:
    """Every literal in dedup_apply.py that is a MATCH ... CREATE query.

    Read through ``ast`` rather than a regex: these queries are written as
    implicitly-concatenated string literals, and only the parser joins them.
    A line-based regex would silently match nothing -- which is exactly how a
    "no CREATE found" assertion can pass for the wrong reason.
    """
    import ast

    with open("janus_graph/pipeline/dedup_apply.py", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "MATCH" in node.value
            and "CREATE" in node.value
        ):
            out.append(node.value)
    return out


def test_move_queries_bind_both_endpoints():
    """Every MATCH ... CREATE must name its MATCH nodes.

    Regression pin for the 2026-10-09 live incident, covering both the MENTIONS
    and the RELATES_TO move: an anonymous node in MATCH does not bind the
    CREATE alias, so CREATE quietly builds a fresh empty node instead.
    """
    queries = _move_queries()
    assert len(queries) >= 2, (
        "expected both the MENTIONS and the RELATES_TO move to be found, got %d" % len(queries)
    )
    for cypher in queries:
        bound = _match_aliases(cypher)
        assert len(bound) >= 2, "MATCH ... CREATE binds %r; both endpoints must be named: %r" % (
            bound,
            cypher,
        )


def test_relations_create_also_binds_both_endpoints():
    """Phase B had the identical shape, so it gets the identical guard.

    relations_to_redirect was 0 on the live run, so this branch never executed
    there -- the latent copy would have shipped silently.
    """
    rel = [c for c in _move_queries() if "RELATES_TO" in c]
    assert rel, "expected the RELATES_TO CREATE to be present"
    for cypher in rel:
        bound = _match_aliases(cypher)
        assert len(bound) >= 2, "RELATES_TO CREATE does not bind both endpoints: %r" % (cypher,)
