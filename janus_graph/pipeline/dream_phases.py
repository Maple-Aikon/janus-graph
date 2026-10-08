"""Dream Mode phase helpers — pure logic, no I/O, no FalkorDB import.

Separated from :mod:`janus_graph.pipeline.dream` (orchestration) so that what a
phase *decides* is unit-testable without a live database, exactly like
:mod:`janus_graph.pipeline.dedup`.

Phase 1 (community clustering) — what was actually wrong
-------------------------------------------------------
``dream.bounded_label_propagation`` exists and is exercised by the test suite,
but it is **not deterministic**: run it on identical data with a different dict
insertion order and the partition changes. Measured on the 2026-10-07 snapshot
(5,764 entities / 12,295 RELATES_TO):

    baseline key order ....... 2,796 communities
    reversed key order ....... 1,210 communities
    shuffled seed=1/7/42 ..... 1,205 / 1,736 / 1,783 communities

There are TWO causes, and only the first was diagnosed at the time:

1. **A schedule problem, not a bound problem.** The update is *asynchronous and
   in-place* — a node's new label is visible to its neighbours later in the same
   sweep — and the seed is the dict's insertion order. Three hypotheses, tested
   before writing code:

   H1 "only the schedule and seed cause it, not the algorithm itself" —
   **SURVIVED.** A synchronous update seeded from a lexicographic total order
   produced ONE partition hash across baseline plus four shuffles
   (seeds 1/7/42/1337, shuffling both node order and neighbour order).
   H2 "label propagation converges, so run it to convergence" —
   **FALSIFIED.** On the real snapshot it never converges: it enters a 2-cycle at
   round 21 with 1,024 oscillating nodes. Convergence is not available here, so
   cycle detection plus a fixed resolution rule is mandatory.
   H3 "it was merely stopping early at ``max_iter``" — **FALSIFIED.** Convergence
   would need 21 rounds, one past the cap of 20, so the cap was a symptom.

2. **The number it yields is not a measurement.** Even fully deterministic, the
   result is dominated by structure no label-propagation variant here can see:
   the graph has 994 connected components and the largest holds 4,592 nodes
   (79.7%). Plain LPA provably cannot split a connected component, so a giant
   community is expected rather than a bug. Pinning low-degree nodes moves the
   headline count from 1,296 to 4,891 without adding any fact about the graph.

So Phase 1 reports the partition **together with its structural floor** (component
count, largest component, whether the largest community is bigger than any
component could explain) so the number is readable for what it is, instead of
printing a bare count that looks like a measurement.

Phase 3 (orphan pruning) — what "safe" means
----------------------------------------------
An orphan has no MENTIONS and no RELATES_TO pointing at it. That makes it
*unreferenced*, not *disposable*: the 10 orphans in the snapshot carry 84-251
chars of summary text each (1,326 chars total), e.g. ``stable diffusion`` —
"No stable diffusion references or integration found in the P..." — a negative
finding that exists precisely BECAUSE nothing links to it.

So this module never deletes on unreferencedness alone. It deletes only nodes
that are unreferenced **and** carry no summary text, and reports everything else
as a candidate for a human. On the 2026-10-07 snapshot that yields 10 candidates
and **0 deletions** — which is the correct answer, not a failure.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

__all__ = [
    "ClusterResult",
    "Orphan",
    "PrunePlan",
    "build_adjacency",
    "cluster_graph",
    "connected_components",
    "count_low_degree",
    "find_orphan_candidates",
    "partition_hash",
    "plan_orphan_pruning",
    "plan_clustering",
    "referenced_uuids",
    "summarize_prune_plan",
]

_WORD = re.compile(r"\S")

#: Nodes whose degree is below this are PINNED to their own label instead of
#: adopting a neighbour's. Plain label propagation cannot cut a connected
#: component, and this graph has one component holding 4,592 of 5,764 nodes
#: (79.7%) — measured 2026-10-07. Without pinning, every degree-1 leaf follows
#: its single neighbour and the "community count" is just the component count
#: wearing a lab coat. Measured: min_degree 0/2/3/5 -> 1,296 / 3,225 / 4,134 /
#: 4,891 communities, i.e. the headline number tracks this knob.
DEFAULT_MIN_DEGREE = 3


@dataclass
class Orphan:
    """One unreferenced entity, with the reason it was or was not prunable."""

    uuid: str
    name: str
    summary: str = ""
    #: False when the node is unreferenced but carries text we must not lose.
    prunable: bool = False
    reason: str = ""

    @property
    def summary_chars(self) -> int:
        return len(self.summary)


@dataclass
class PrunePlan:
    """What Phase 3 would do. Deciding is free; applying is not."""

    candidates: List[Orphan] = field(default_factory=list)
    deleted: int = 0
    #: Nodes skipped because they hold text. These are the interesting output.
    protected: int = 0
    protected_chars: int = 0

    @property
    def prunable(self) -> List[Orphan]:
        return [o for o in self.candidates if o.prunable]

    def describe(self) -> str:
        return (
            "candidates=%d prunable=%d protected=%d (holding %d chars), deleted=%d"
            % (
                len(self.candidates),
                len(self.prunable),
                self.protected,
                self.protected_chars,
                self.deleted,
            )
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidates": len(self.candidates),
            "prunable": len(self.prunable),
            "protected": self.protected,
            "protected_chars": self.protected_chars,
            "deleted": self.deleted,
            "nodes": [
                {
                    "uuid": o.uuid,
                    "name": o.name,
                    "summary_chars": o.summary_chars,
                    "prunable": o.prunable,
                    "reason": o.reason,
                }
                for o in self.candidates
            ],
        }


def _as_list(value: Any) -> List[Any]:
    """Rows arrive as lists from some drivers and tuples from others."""
    if value is None:
        return []
    return list(value)


def referenced_uuids(
    mentions: Iterable[Dict[str, Any]],
    relations: Iterable[Dict[str, Any]],
) -> Set[str]:
    """Every uuid that some edge endpoint points at.

    RELATES_TO carries two spellings for the same thing (``source_node_uuid``
    /``target_node_uuid`` and ``src_uuid``/``dst_uuid``); both are read so a
    spelling change cannot silently redefine an orphan as unreferenced.
    """
    out: Set[str] = set()
    for m in mentions:
        for key in ("dst_uuid",):
            v = m.get(key)
            if v:
                out.add(str(v))
    for r in relations:
        for key in (
            "source_node_uuid",
            "target_node_uuid",
            "src_uuid",
            "dst_uuid",
        ):
            v = r.get(key)
            if v:
                out.add(str(v))
    return out


def find_orphan_candidates(
    entities: Iterable[Dict[str, Any]],
    mentions: Iterable[Dict[str, Any]] = (),
    relations: Iterable[Dict[str, Any]] = (),
) -> List[Orphan]:
    """Entities no edge points at, annotated with whether they are prunable."""
    refs = referenced_uuids(mentions, relations)
    out: List[Orphan] = []
    for e in entities:
        uuid = e.get("uuid")
        if uuid is None:
            continue
        uuid = str(uuid)
        if uuid in refs:
            continue
        summary = e.get("summary") or ""
        name = e.get("name") or ""
        if summary.strip():
            out.append(
                Orphan(
                    uuid=uuid,
                    name=str(name),
                    summary=summary,
                    prunable=False,
                    reason="unreferenced but holds %d chars of summary"
                    % len(summary),
                )
            )
        else:
            out.append(
                Orphan(
                    uuid=uuid,
                    name=str(name),
                    summary="",
                    prunable=True,
                    reason="unreferenced and empty",
                )
            )
    return out


def plan_orphan_pruning(
    entities: Iterable[Dict[str, Any]],
    mentions: Iterable[Dict[str, Any]] = (),
    relations: Iterable[Dict[str, Any]] = (),
) -> PrunePlan:
    """Build the Phase 3 plan. Never touches a database."""
    cands = find_orphan_candidates(entities, mentions, relations)
    plan = PrunePlan(candidates=cands)
    plan.protected = sum(1 for o in cands if not o.prunable)
    plan.protected_chars = sum(o.summary_chars for o in cands if not o.prunable)
    return plan


def summarize_prune_plan(plan: PrunePlan) -> str:
    """One line suitable for a dream report."""
    return plan.describe()

# ---------------------------------------------------------------------------
# Phase 1 — deterministic community clustering
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    """A partition of the entity graph, plus what it does and does not mean.

    ``partition_hash`` is the reproducible artifact. ``communities`` alone is
    not safe to report as a measurement: it moves with ``min_degree`` (1,296 ->
    4,891 on the 2026-10-07 snapshot) while the graph stays identical.
    ``oversized`` is the honest caveat — when the largest community is bigger
    than the largest connected component could explain, the partition has
    merged genuinely separate structure rather than discovered communities.
    """

    nodes: int
    communities: int
    rounds: int
    converged: bool
    cycle_detected: bool
    oscillating_nodes: int
    pinned_nodes: int
    components: int
    largest_component: int
    largest_community: int
    min_degree: int
    partition_hash: str
    oversized: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "communities": self.communities,
            "rounds": self.rounds,
            "converged": self.converged,
            "cycle_detected": self.cycle_detected,
            "oscillating_nodes": self.oscillating_nodes,
            "pinned_nodes": self.pinned_nodes,
            "components": self.components,
            "largest_component": self.largest_component,
            "largest_community": self.largest_community,
            "min_degree": self.min_degree,
            "partition_hash": self.partition_hash,
            "oversized": self.oversized,
        }

    def describe(self) -> str:
        if not self.nodes:
            return "SKIPPED (no entities in the snapshot)"
        parts = [
            "%d communities over %d nodes" % (self.communities, self.nodes),
            "deterministic" if not self.cycle_detected
            else "deterministic-after-cycle-resolution",
            "rounds=%d" % self.rounds,
            "structural floor: %d components, largest %d"
            % (self.components, self.largest_component),
            "hash=%s" % self.partition_hash,
        ]
        if self.pinned_nodes:
            parts.append(
                "%d low-degree nodes pinned (min_degree=%d)"
                % (self.pinned_nodes, self.min_degree)
            )
        if not self.converged:
            parts.append("did not converge in %d rounds" % self.rounds)
        if self.cycle_detected:
            parts.append(
                "2-cycle on %d nodes resolved by smallest-uuid rule"
                % self.oscillating_nodes
            )
        if self.oversized:
            parts.append(
                "CAVEAT largest community %d exceeds the largest connected "
                "component %d, so this count is not a semantic clustering"
                % (self.largest_community, self.largest_component)
            )
        return "; ".join(parts)


def build_adjacency(
    entities: Iterable[Dict[str, Any]],
    relations: Iterable[Dict[str, Any]] = (),
) -> Dict[str, Set[str]]:
    """Undirected adjacency over entity uuids, from the phase-2 snapshot.

    Edges pointing outside ``entities`` are dropped rather than creating ghost
    nodes: the snapshot is the authority on which nodes exist.
    """
    nodes: Set[str] = set()
    for ent in entities:
        nodes.add(str(ent.get("uuid")))
    adj: Dict[str, Set[str]] = {u: set() for u in nodes}
    for rel in relations:
        src = str(rel.get("src_uuid"))
        dst = str(rel.get("dst_uuid"))
        if src not in nodes or dst not in nodes or src == dst:
            continue
        adj[src].add(dst)
        adj[dst].add(src)
    return adj


def count_low_degree(adj: Dict[str, Set[str]], min_degree: int) -> int:
    """How many nodes are pinned because their degree is below ``min_degree``."""
    return sum(1 for u, vs in adj.items() if len(vs) < min_degree)


def connected_components(adj: Dict[str, Set[str]]) -> List[List[str]]:
    """Connected components, largest first, each sorted.

    The structural floor: no label-propagation variant can split a connected
    component, so this is the number a Phase 1 result must be read against.
    """
    seen: Set[str] = set()
    comps: List[List[str]] = []
    for root in sorted(adj):
        if root in seen:
            continue
        stack = [root]
        seen.add(root)
        comp: List[str] = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in sorted(adj[node]):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: (-len(c), c[0] if c else ""))
    return comps


def partition_hash(labels: Dict[str, int]) -> str:
    """Stable fingerprint of a PARTITION, independent of label numbering.

    Label ids are an internal artifact — the same partition reached by a
    different route may number its groups differently — so the hash is taken
    over sorted member uuids per group, not over the labels themselves.
    """
    groups: Dict[int, List[str]] = {}
    for node, label in labels.items():
        groups.setdefault(label, []).append(node)
    canonical = sorted(tuple(sorted(members)) for members in groups.values())
    # LENGTH-PREFIXED, but the prefix is defence-in-depth, NOT a fix for a
    # bug I can demonstrate. The comment here used to claim that joining with
    # a bare separator collides -- [('a','b')] and [('a',),('b',)] rendering
    # as "a<sep>b". I could not reproduce that: my checker compared groups
    # WITHIN a single partition, so it could never see the cross-partition
    # collision the claim is about. Redone exhaustively (Bell-number
    # recursion, every partition of n<=6 tokens) the bare join collides 0
    # times -- for uuid-shaped tokens (the real caller) and for mixed-length
    # ones alike. A bare join only breaks if a token CONTAINS the separator,
    # which is the very assumption the prefix would be replacing. So the
    # prefix stays (it costs nothing) but its presence is not evidence that
    # a hash collision was ever found and fixed here.
    blob = "".join(
        "%d:%s" % (len(group), chr(31).join(group)) for group in canonical
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _resolve_cycle(
    labels: Dict[str, int],
    oscillating: Iterable[str],
    comp_of: Optional[Dict[str, int]] = None,
) -> None:
    """Collapse an LPA 2-cycle onto ONE label per component, in place.

    At the moment the cycle is detected, the oscillating nodes carry DISTINCT
    labels -- on a graph of two disjoint edges the labels at cycle time are
    {a:0, b:1, c:2, d:3} with all four nodes oscillating. So grouping by
    ``(component, label)`` yields singletons and assigns each node the value it
    already holds: a no-op, which is what an earlier version of this function
    did silently.

    The oscillating set is therefore grouped by COMPONENT alone. Within one
    component every oscillating node adopts the label of the component's
    smallest-uuid oscillator, which is a fixed point of the 2-cycle: the next
    sweep reads a single label across the set and the partition stops moving.

    Grouping by component also keeps separate components separate. Collapsing
    the whole oscillating set at once merged three disjoint 2-node chains into
    one 6-node community, so ``largest_community`` exceeded
    ``largest_component`` and the partition described a graph that does not
    exist.
    """
    members = sorted(set(oscillating))
    if not members:
        return

    if comp_of is None:
        comp_of = {node: 0 for node in members}

    by_component: Dict[Any, List[str]] = {}
    for node in members:
        by_component.setdefault(comp_of.get(node, 0), []).append(node)

    for group in by_component.values():
        keeper_label = labels[group[0]]      # members are sorted => smallest uuid
        for node in group:
            labels[node] = keeper_label


def cluster_graph(
    adj: Dict[str, Set[str]],
    min_degree: int = DEFAULT_MIN_DEGREE,
    max_rounds: int = 1000,
) -> ClusterResult:
    """Deterministic synchronous label propagation.

    Determinism comes from three places, each necessary:

    * nodes are processed in sorted order, so the sweep is a total order;
    * the seed label is the node's rank in that order, not insertion order;
    * ties break on the SMALLEST label, so no tie is resolved by arrival.

    The update is **synchronous** — every node reads the previous round's
    labels — which is what makes each round a pure function of the graph.

    Cycle handling: label propagation provably oscillates on bipartite-ish
    regions (measured: 1,024 oscillating nodes on the 2026-10-07 snapshot). The
    2-cycle is detected and resolved by :func:`_resolve_cycle`, so the result
    never depends on where the sweep would otherwise have stopped.
    """
    # Normalise first, and make it SYMMETRIC. Two callers describing the same
    # undirected graph differently -- one listing both endpoints, one listing
    # each neighbour only under its own key -- produced 1 community and 2
    # communities respectively. That is exactly the shape-dependence this
    # module exists to remove, re-entering through the front door: unioning
    # the keys is not enough, the missing reverse edges must be added.
    raw = {u: set(vs) for u, vs in adj.items()}
    keys = set(raw)
    keys |= {nb for vs in raw.values() for nb in vs}
    adj = {u: set() for u in keys}
    for u, vs in raw.items():
        if u not in keys:
            continue
        for v in vs:
            if v in keys and u != v:
                adj[u].add(v)
                adj[v].add(u)          # <-- the symmetry that was missing

    nodes = sorted(adj)
    if not nodes:
        return ClusterResult(
            nodes=0, communities=0, rounds=0, converged=True,
            cycle_detected=False, oscillating_nodes=0, pinned_nodes=0,
            components=0, largest_component=0, largest_community=0,
            min_degree=min_degree, partition_hash=partition_hash({}),
            oversized=False,
        )

    pinned = {u for u in nodes if len(adj[u]) < min_degree}
    rank = {u: i for i, u in enumerate(nodes)}
    labels = {u: rank[u] for u in nodes}

    comps = connected_components(adj)
    comp_of = {node: idx
              for idx, comp in enumerate(comps) for node in comp}

    prev = frozenset(labels.items())
    seen = {prev}
    oscillating: List[str] = []
    cycle = False
    converged = False
    rounds = 0

    for rounds in range(1, max_rounds + 1):
        new: Dict[str, int] = {}
        for node in nodes:
            if node in pinned:
                new[node] = labels[node]
                continue
            tally: Dict[int, int] = {}
            for nb in adj[node]:
                if nb not in pinned:
                    lab = labels[nb]
                    tally[lab] = tally.get(lab, 0) + 1
            if not tally:
                new[node] = labels[node]
            else:
                new[node] = min(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        labels = new
        current = frozenset(labels.items())
        if current == prev:
            converged = True
            break
        if current in seen:
            prev_map = dict(prev)
            oscillating = [u for u in nodes if labels[u] != prev_map[u]]
            cycle = True
            break
        seen.add(current)
        prev = current

    if oscillating:
        # Per-component: collapsing the whole oscillating set merged
        # separate components into one community (measured on three
        # disjoint 2-node chains: 1 community spanning 6 nodes).
        _resolve_cycle(labels, oscillating, comp_of)

    sizes: Dict[int, int] = {}
    for label in labels.values():
        sizes[label] = sizes.get(label, 0) + 1
    largest_community = max(sizes.values()) if sizes else 0
    largest_component = len(comps[0]) if comps else 0

    return ClusterResult(
        nodes=len(nodes),
        communities=len(sizes),
        rounds=rounds,
        converged=converged,
        cycle_detected=cycle,
        oscillating_nodes=len(oscillating),
        pinned_nodes=len(pinned),
        components=len(comps),
        largest_component=largest_component,
        largest_community=largest_community,
        min_degree=min_degree,
        partition_hash=partition_hash(labels),
        oversized=largest_community > largest_component,
    )


def plan_clustering(
    entities: Iterable[Dict[str, Any]],
    relations: Iterable[Dict[str, Any]] = (),
    min_degree: int = DEFAULT_MIN_DEGREE,
) -> ClusterResult:
    """Phase 1 entry point. Never touches a database."""
    return cluster_graph(
        build_adjacency(entities, relations), min_degree=min_degree
    )


def cluster_report(
    result: ClusterResult,
    total_episodes: int,
    threshold: int,
    force: bool,
) -> str:
    """The Phase 1 line: the gate decision AND the partition that was computed.

    Both halves are reported because each was previously reported alone. An
    earlier version printed ``DONE`` after merely counting episodes, which is a
    different claim from "clustered the graph".
    """
    gate = "forced" if force else "gated"
    if force:
        # force short-circuits the comparison, so with 0 episodes an inequality
        # printed here would read as true when no comparison happened at all.
        why = "force=True bypassed the %d-episode threshold (%d episodes)" % (
            threshold, total_episodes,
        )
    elif total_episodes >= threshold:
        why = "gate passed (%d episodes >= %d)" % (total_episodes, threshold)
    else:
        why = "gate closed (%d episodes < %d)" % (total_episodes, threshold)
    if not result.nodes:
        # An empty snapshot is not a DONE. Wrapping it in DONE produced
        # "DONE (forced, ...): SKIPPED (no entities in the snapshot)", which is
        # self-contradictory and would read to a report parser as real work.
        return "SKIPPED (%s, %s): %s" % (gate, why, result.describe())
    return "DONE (%s, %s): %s" % (gate, why, result.describe())
