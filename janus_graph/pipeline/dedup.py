"""Phase 2 dedup planning — pure logic, no I/O, no FalkorDB import.

Separated from :mod:`janus_graph.pipeline.dream` so the decision of WHAT to
merge is unit-testable without a live database. Execution (Cypher against
FalkorDB) lives in :mod:`janus_graph.pipeline.dedup_apply`.

Why the merge key is (name AND summary-similarity) and not name alone
--------------------------------------------------------------------------
Measured on the 2026-10-07 snapshot of ``graphiti_memory`` (309 duplicate-name
clusters / 720 nodes): **0 of 309 clusters had identical summaries**, and the
highest within-cluster summary Jaccard was 0.882. Graphiti deliberately creates
one node per distinct aspect of a name and links them with RELATES_TO, so
merging by name alone would collapse distinct knowledge and drop ~48% of the
summary text in the affected nodes.

Consequently the default threshold is deliberately conservative: it merges
only nodes whose summaries are near-restatements of each other.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = [
    "DEFAULT_DEDUP_THRESHOLD",
    "DedupPlan",
    "escape_cypher_string",
    "jaccard",
    "plan_dedup",
    "tokenize",
]

#: Minimum word-token Jaccard between two summaries before a merge is allowed.
#:
#: The authoritative value is ``DreamConfig.dedup_threshold`` (janus_graph/config.py);
#: read it through ``dedup_apply._dedup_threshold(cfg)``. This constant exists only
#: as the fallback for callers that pass no threshold — ``contracts.DreamSettings``
#: also declares one (0.85), but that class is unreachable from the pipeline: it
#: had zero readers while the planner used a literal, so the two silently
#: disagreed.
#:
#: 0.80 keeps 2 of 309 clusters on the 2026-10-07 snapshot; both were verified by
#: hand to be genuine near-restatements. 0.85 keeps 1 (it drops CRG_TOOLS at
#: similarity 0.8095), and 0.90 keeps 0.
DEFAULT_DEDUP_THRESHOLD = 0.80

_WORD = re.compile(r"[a-z0-9_]+")


def tokenize(text: Optional[str]) -> frozenset:
    """Lowercase word tokens used for the similarity test.

    Underscore is kept inside tokens on purpose: this graph is full of code
    identifiers (``step14_bare_xml_params``, ``get_github_tags_sorted``) and
    splitting on it would make ``a_b`` and ``b_a`` look identical.
    """
    return frozenset(_WORD.findall((text or "").lower()))


def jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity. Two empty summaries count as identical (1.0)."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def escape_cypher_string(value: str) -> str:
    """Escape a string for single-quoted openCypher literals.

    This FalkorDB build escapes with a BACKSLASH, not SQL-style doubled quotes.
    Order matters: the backslash itself must be escaped FIRST, otherwise the
    backslashes introduced by quote-escaping get doubled.

    Verified against the 2026-10-01 escaping work; see the parallel test
    ``test_escape_cypher_string_matches_verified_cases`` which pins 12 exact
    cases plus negative controls.
    """
    out = (value or "").replace("\\", "\\\\")
    out = out.replace("'", "\\'")
    return out


@dataclass
class Merge:
    """One planned merge: ``victim_uuid`` folds into ``survivor_uuid``."""

    name: str
    survivor_uuid: str
    victim_uuid: str
    similarity: float
    survivor_summary_len: int
    victim_summary_len: int


@dataclass
class DedupPlan:
    """The full outcome of planning, with enough detail to explain itself."""

    merges: List[Merge] = field(default_factory=list)
    clusters_scanned: int = 0
    duplicate_name_nodes: int = 0
    mentions_to_redirect: int = 0
    relations_to_redirect: int = 0
    #: Duplicate edges that collapse onto an identical (src,dst,fact) triple
    #: AFTER redirection. EXCLUDES self-loops, which have their own counter --
    #: a self-loop is a different kind of loss (an edge whose two endpoints
    #: became one node) than a duplicate (two copies of the same fact), and
    #: merging them into one number would hide which happened.
    mentions_to_drop: int = 0
    relations_to_drop: int = 0
    #: Edges whose BOTH endpoints folded into one node and were removed.
    self_loops_dropped: int = 0
    nodes_to_delete: int = 0
    summary_chars_saved: int = 0
    threshold: float = DEFAULT_DEDUP_THRESHOLD

    @property
    def is_empty(self) -> bool:
        return not self.merges

    @property
    def total_relations_removed(self) -> int:
        """All RELATES_TO losses, self-loops included."""
        return self.relations_to_drop + self.self_loops_dropped

    def as_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "clusters_scanned": self.clusters_scanned,
            "duplicate_name_nodes": self.duplicate_name_nodes,
            "merges": len(self.merges),
            "nodes_to_delete": self.nodes_to_delete,
            "mentions_to_redirect": self.mentions_to_redirect,
            "relations_to_redirect": self.relations_to_redirect,
            "mentions_to_drop": self.mentions_to_drop,
            "relations_to_drop": self.relations_to_drop,
            "self_loops_dropped": self.self_loops_dropped,
            "relations_removed_total": self.total_relations_removed,
            "summary_chars_saved": self.summary_chars_saved,
        }

    def describe(self) -> str:
        """The operator-facing one-liner stored in ``phase_2_deduplication``.

        Must never read as bare "DONE" -- that literal is what made the
        original no-op indistinguishable from real work.
        """
        head = (
            "DONE (scanned={clusters_scanned} dup-name clusters / "
            "{duplicate_name_nodes} nodes, merged={merges}, removed={nodes_to_delete} "
            "nodes, moved={mentions_to_redirect} MENTIONS + {relations_to_redirect} "
            "RELATES_TO, dropped={mentions_to_drop} MENTIONS + "
            "{relations_removed_total} RELATES_TO "
            "(={relations_to_drop} dup + {self_loops_dropped} self-loop), "
            "saved={summary_chars_saved} summary chars, threshold={threshold})"
        )
        return head.format(**self.as_dict())


def _pick_survivor(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Choose which node of a mergeable pair survives.

    Policy, in order:
      1. Longest summary — never discards text the loser's summary already had.
      2. Earliest ``created_at`` — deterministic tie-break, stable over time.
      3. Lowest ``uuid`` — final tie-break so replanning is idempotent.

    Rationale is measured, not aesthetic: on the 2026-10-07 snapshot a
    "first created wins" rule would have dropped summary text that the
    "longest wins" rule keeps.
    """
    return sorted(
        nodes,
        key=lambda n: (
            -len(n.get("summary") or ""),
            str(n.get("created_at") or ""),
            str(n.get("uuid") or ""),
        ),
    )[0]


def plan_dedup(
    entities: Iterable[Dict[str, Any]],
    mentions: Iterable[Dict[str, Any]] = (),
    relations: Iterable[Dict[str, Any]] = (),
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> DedupPlan:
    """Plan duplicate merges without touching the database.

    Args:
        entities: dicts with at least ``uuid``, ``name``, ``summary``.
        mentions: dicts with ``src_uuid``/``dst_uuid`` (Episodic -> Entity).
        relations: dicts with ``src_uuid``/``dst_uuid`` (Entity -> Entity);
            the duplicated ``source_node_uuid``/``target_node_uuid`` fields are
            rewritten too, since both spellings exist on RELATES_TO.
        threshold: minimum summary Jaccard. Higher = more conservative.

    The plan is a *chain*: after folding A into B, B is re-tested against the
    rest of the cluster, so a cluster of three near-identical summaries
    collapses to one survivor rather than two.
    """
    plan = DedupPlan(threshold=threshold)

    by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ent in entities:
        by_name[ent.get("name") or ""].append(ent)

    victims: Dict[str, str] = {}  # victim_uuid -> survivor_uuid
    survivors: Dict[str, str] = {}  # survivor_uuid -> name

    for name, nodes in by_name.items():
        if len(nodes) < 2:
            continue
        plan.clusters_scanned += 1
        plan.duplicate_name_nodes += len(nodes)

        remaining = list(nodes)
        while len(remaining) > 1:
            best: Optional[Tuple[float, Dict[str, Any], Dict[str, Any]]] = None
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    score = jaccard(
                        tokenize(remaining[i].get("summary")),
                        tokenize(remaining[j].get("summary")),
                    )
                    if score >= threshold and (best is None or score > best[0]):
                        best = (score, remaining[i], remaining[j])
            if best is None:
                break
            score, a, b = best
            # _pick_survivor returns the survivor; the other one is the victim.
            keeper = _pick_survivor([a, b])
            loser = b if keeper is a else a
            keeper_uuid = str(keeper.get("uuid"))
            loser_uuid = str(loser.get("uuid"))
            if keeper_uuid == loser_uuid:
                break
            victims[loser_uuid] = keeper_uuid
            survivors[keeper_uuid] = name
            plan.merges.append(
                Merge(
                    name=name,
                    survivor_uuid=keeper_uuid,
                    victim_uuid=loser_uuid,
                    similarity=round(score, 4),
                    survivor_summary_len=len(keeper.get("summary") or ""),
                    victim_summary_len=len(loser.get("summary") or ""),
                )
            )
            remaining = [n for n in remaining if str(n.get("uuid")) != loser_uuid]

    plan.nodes_to_delete = len(victims)

    # Resolve chains: A->B, B->C must collapse to A->C.
    def _root(uuid: str) -> str:
        seen = set()
        cur = uuid
        while cur in victims and cur not in seen:
            seen.add(cur)
            cur = victims[cur]
        return cur

    redirect = {v: _root(v) for v in victims}

    keeper_out_edges = defaultdict(set)
    for rel in relations:
        src, dst = str(rel.get("src_uuid") or ""), str(rel.get("dst_uuid") or "")
        src, dst = redirect.get(src, src), redirect.get(dst, dst)
        keeper_out_edges[src].add(dst)

    seen_mentions = set()
    for men in mentions:
        src = str(men.get("src_uuid") or "")
        dst = redirect.get(str(men.get("dst_uuid") or ""), str(men.get("dst_uuid") or ""))
        key = (src, dst)
        if key in seen_mentions:
            plan.mentions_to_drop += 1
            continue
        seen_mentions.add(key)

    seen_rel = set()
    for rel in relations:
        src, dst = str(rel.get("src_uuid") or ""), str(rel.get("dst_uuid") or "")
        new_src, new_dst = redirect.get(src, src), redirect.get(dst, dst)
        if new_src == new_dst:
            plan.relations_to_drop += 1
            continue
        key = (new_src, new_dst, str(rel.get("fact") or ""))
        if key in seen_rel:
            plan.relations_to_drop += 1
            continue
        seen_rel.add(key)

    plan.mentions_to_redirect = sum(
        1
        for men in mentions
        if str(men.get("dst_uuid") or "") in redirect
    )
    plan.relations_to_redirect = sum(
        1
        for rel in relations
        if str(rel.get("src_uuid") or "") in redirect
        or str(rel.get("dst_uuid") or "") in redirect
    )

    # Counter-semantics correction (2026-10-07, measured on the snapshot).
    # An earlier draft walked the WHOLE graph to count drops, so the numbers
    # included redundancy that already existed before this merge: the snapshot
    # held 136 duplicate MENTIONS pairs and 12 duplicate RELATES_TO triples with
    # zero merges applied. Reporting those as "dropped by dedup" credits the
    # phase with cleanup it did not perform.
    #
    # A second, subtler bug followed: counting only the edges pointing AT a
    # victim cannot detect a collision, because the colliding partner is the
    # survivor's own untouched edge. So each destination is counted over ALL
    # its edges (redirected or not), and a collision is charged to this merge
    # only when the destination is one this merge actually touched.
    def _collisions(edges, keyfn, touched):
        groups: Dict[Any, List[Any]] = defaultdict(list)
        for edge in edges:
            groups[keyfn(edge)].append(edge)
        drops = 0
        for key, group in groups.items():
            if len(group) < 2:
                continue
            hit = False
            for edge in group:
                if touched(edge):
                    hit = True
                    break
            if not hit:
                continue  # redundancy this merge did not create
            drops += len(group) - 1
        return drops

    def _ment_key(men):
        dst = str(men.get("dst_uuid") or "")
        return (str(men.get("src_uuid") or ""), redirect.get(dst, dst))

    def _ment_touched(men):
        return str(men.get("dst_uuid") or "") in redirect

    plan.mentions_to_drop = _collisions(mentions, _ment_key, _ment_touched)

    def _rel_key(rel):
        src = str(rel.get("src_uuid") or "")
        dst = str(rel.get("dst_uuid") or "")
        new_src, new_dst = redirect.get(src, src), redirect.get(dst, dst)
        # src == dst means the edge folded onto itself and is dropped outright.
        if new_src == new_dst:
            return ("__self_loop__", src, dst)
        return (new_src, new_dst, str(rel.get("fact") or ""))

    def _rel_touched(rel):
        return (
            str(rel.get("src_uuid") or "") in redirect
            or str(rel.get("dst_uuid") or "") in redirect
        )

    self_loops = 0
    for rel in relations:
        if not _rel_touched(rel):
            continue
        src = str(rel.get("src_uuid") or "")
        dst = str(rel.get("dst_uuid") or "")
        if redirect.get(src, src) == redirect.get(dst, dst):
            self_loops += 1
    plan.self_loops_dropped = self_loops
    plan.relations_to_drop = _collisions(relations, _rel_key, _rel_touched)
    plan.summary_chars_saved = sum(m.victim_summary_len for m in plan.merges)

    return plan
