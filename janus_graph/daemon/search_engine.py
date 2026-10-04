"""Phase 3.1 SearchEngine — BFS traversal + cosine rerank + MMR.

Pipeline (per Plan #2 §Phase 3.1):
  1. Validate seeds exist via get_entity(name) — fast path.
  2. Cypher BFS: UNWIND seeds + MATCH path [*1..max_hops] + invalid_at IS
     NULL + edge_types filter + LIMIT 5000 (hard cap before rerank).
  3. Extract (fact_text, source_entity, target_entity, edge_type, hop,
     path) tuples.
  4. Cosine via EmbeddingClient (semaphore-bounded batch).
  5. Filter: drop cosine < min_cosine OR invalid_at IS NOT NULL.
  6. MMR rerank (mmr_lambda=0.7 default) with top_k=limit.
  7. Return JSON payload with degraded flag.

Errors map to HTTP codes at the handler layer:
  - SEED_NOT_FOUND (404), INVALID_HOPS / INVALID_SEED / INVALID_LIMIT (400),
    MIN_COSINE_OUT_OF_RANGE (422), FALKOR_DOWN (503), TRAVERSAL_TIMEOUT (504).

Anti-patterns (per Plan §Phase 3.1 notes):
  - Do NOT mix BFS into ``search_memory`` (semantic MMR rerank != structural
    traversal; mixing degrades both). This is a separate endpoint, separate
    config keys, separate pipeline.
  - Do NOT ship the minimal BFS helper then duplicate in Phase 4 daemon —
    this module IS the canonical implementation; Phase 4 will import it.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import redis.asyncio as redis_async

from ..config import EngineConfig
from .embedding_client import EmbeddingClient
from .phase3_settings import DaemonSearchGraphSettings

logger = logging.getLogger("janus_graph.daemon.search_engine")


# ─── domain types ──────────────────────────────────────────────────────


@dataclass
class FactRow:
    """A single fact extracted from a BFS path."""

    fact: str
    source_entity: str
    target_entity: str
    edge_type: str
    hop_distance: int
    path: List[str]
    cosine: Optional[float] = None
    invalid_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact": self.fact,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "edge_type": self.edge_type,
            "hop_distance": self.hop_distance,
            "path": self.path,
            "cosine": self.cosine,
            "invalid_at": self.invalid_at,
        }


@dataclass
class SearchResponse:
    """Structured result of SearchEngine.search()."""

    results: List[FactRow] = field(default_factory=list)
    count: int = 0
    elapsed_ms: int = 0
    degraded: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "count": self.count,
            "elapsed_ms": self.elapsed_ms,
            "degraded": self.degraded,
            "warnings": self.warnings,
        }


class SearchValidationError(Exception):
    """Schema validation error — mapped to HTTP 400/422."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class SearchBackendError(Exception):
    """Falkor/embed backend error — mapped to HTTP 503."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


# ─── engine ────────────────────────────────────────────────────────────


# Cypher BFS query — uses UNWIND over an inlined seed list, MATCH path
# [*1..max_hops], filters by rel.invalid_at IS NULL (no-op on current KG
# schema which doesn't have invalid_at on rels; filter is a forward-compat
# for when Phase 4 graphiti ingest adds it).
#
# Schema note (RE-VERIFIED 2026-10-04 against graphiti_memory, live counts):
#  - KG has only one rel type: MENTIONS (rel props: uuid, group_id, created_at)
#    plus a ``fact`` property added by the graphiti ingest.
#  - 32,862 rels total; 10,686 carry ``r.fact``; 0 carry ``r.summary``.
#  - 9,805 nodes; 5,404 carry ``n.summary``; 0 carry ``n.fact``.
#  - Facts therefore live on the RELATIONSHIP, not on the node. The older
#    note claiming "facts are the target node's summary" was true when
#    written (2026-09-07) and was superseded by the graphiti ingest.
#
# 2026-10-04 BUGFIX — why this projects r.fact, one row per rel:
#  The previous form returned ONE row per PATH and read
#  ``[n IN ns | n.summary][0]`` — index 0 is the SEED node, not the
#  destination — so every expansion of a seed repeated the seed's own
#  summary. A 200-row response could hold exactly ONE distinct fact.
#  Measured live before the fix: 200 rows -> 1 distinct fact.
#  A second bug compounded it: the path-level ``[r IN rels | ...][0]``
#  always picked the FIRST relationship, so rows past the first carried
#  that same edge's fact again.
#  Fix: UNWIND the path's relationships and emit one row per rel, so each
#  fact belongs to the edge that actually produced it. Measured live on
#  seed 'Maple' at the same hops/limit: 200 rows -> 51 distinct facts.
#
# FalkorDB quirks (verified 2026-09-07 / 2026-10-04 against v8.9.241):
#  - Params are baked into the query as ``CYPHER `key`=value ...`` header
#    (NOT a separate ``params`` arg).
#  - Array params are not supported at runtime — seeds are inlined as
#    Cypher literals (``['Maple Aikon', 'Huế']``) by the caller.
#  - List slice bounds are HALF-OPEN like python: ``ns[0..i]`` EXCLUDES
#    index i, so the path through rel i is ``[n IN ns | n.name][0..i + 1]``.
_CYPHER_BFS = """
UNWIND [{seed_literals}] AS seed_name
MATCH (s {name: seed_name})
MATCH path = (s)-[*1..{max_hops}]-(neighbor)
WHERE ALL(rel IN relationships(path) WHERE rel.invalid_at IS NULL OR true)
WITH relationships(path) AS rels, nodes(path) AS ns, length(path) AS hop_count
UNWIND range(0, size(rels) - 1) AS i
WITH ns, hop_count, i, rels[i] AS r
WITH ns, hop_count, i, r,
  CASE WHEN r.fact IS NOT NULL AND r.fact <> '' THEN r.fact
       WHEN ns[i + 1].summary IS NOT NULL AND ns[i + 1].summary <> '' THEN ns[i + 1].summary
       ELSE null END AS edge_fact
WHERE edge_fact IS NOT NULL
RETURN
  type(r) AS edge_type,
  hop_count AS hop_distance,
  [n IN ns | n.name][0..i + 1] AS path_nodes,
  edge_fact,
  [n IN ns | n.name][0] AS seed_in_path,
  ns[i + 1].name AS target_in_path
LIMIT {cap}
"""


# Minimal seed existence probe — uses the standard node-by-name lookup.
# FalkorDB params are baked INTO the Cypher query as a ``CYPHER`` header
# (not as a separate ``params`` arg). See falkordb.graph.Graph._build_params_header
# for the canonical format: ``CYPHER `key1`=value1 `key2`=value2 <query>``.
_CYPHER_EXISTS = "MATCH (n {name: $name}) RETURN n.name LIMIT 1"


class SearchEngine:
    """BFS+cosine+MMR engine for /search/graph endpoint.

    Owns:
      - redis_async.Redis client (lazy connect; fail-soft on Falkor down)
      - EmbeddingClient (injected; shared lifetime)

    Public surface:
      validate_request(payload) -> None       (raises SearchValidationError)
      search(payload) -> SearchResponse       (raises SearchBackendError)
      close() -> None
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        search_settings: DaemonSearchGraphSettings,
        embedding_client: EmbeddingClient,
    ):
        self._engine = engine_config
        self._settings = search_settings
        self._embed = embedding_client
        self._redis: Optional[redis_async.Redis] = None

    async def start(self) -> None:
        """Lazy connect Redis. Failure is NOT raised — backend probes return
        ``FALKOR_DOWN`` via SearchBackendError so the handler returns 503.
        """
        try:
            self._redis = redis_async.Redis(
                host=self._engine.host,
                port=self._engine.port,
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
            )
            await self._redis.ping()
        except Exception as e:  # pragma: no cover — connection refused etc.
            logger.warning("search engine: redis connect failed: %s", e)
            self._redis = None

    async def close(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:  # pragma: no cover
                pass
            self._redis = None

    # ─── validation ────────────────────────────────────────────────────

    def validate_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Schema check + default fill. Raises SearchValidationError on bad input.

        Returns a fully-populated request dict (seeds list, max_hops,
        min_cosine, mmr_lambda, limit, edge_types, include_episodes).
        """
        if not isinstance(payload, dict):
            raise SearchValidationError("INVALID_BODY", "payload must be a JSON object")

        seeds = payload.get("seed_entities")
        if not isinstance(seeds, list) or not seeds:
            raise SearchValidationError("INVALID_SEED", "seed_entities must be a non-empty list")
        if not all(isinstance(s, str) and s.strip() for s in seeds):
            raise SearchValidationError(
                "INVALID_SEED", "seed_entities entries must be non-empty strings"
            )
        if len(seeds) > 5:
            raise SearchValidationError("INVALID_SEED", "seed_entities supports at most 5 entries")

        max_hops = payload.get("max_hops", 2)
        # ``bool`` is a subclass of ``int`` in Python, so a JSON ``true``
        # would otherwise slip through as hops=1. Reject it explicitly.
        if (
            not isinstance(max_hops, int)
            or isinstance(max_hops, bool)
            or max_hops < 1
            or max_hops > 3
        ):
            raise SearchValidationError("INVALID_HOPS", "max_hops must be int in [1, 3]")

        min_cosine = payload.get("min_cosine", self._settings.default_min_cosine)
        if (
            not isinstance(min_cosine, (int, float))
            or isinstance(min_cosine, bool)
            or not (0.0 <= min_cosine <= 1.0)
        ):
            raise SearchValidationError(
                "MIN_COSINE_OUT_OF_RANGE", "min_cosine must be a float in [0.0, 1.0]"
            )

        mmr_lambda = payload.get("mmr_lambda", self._settings.default_mmr_lambda)
        if (
            not isinstance(mmr_lambda, (int, float))
            or isinstance(mmr_lambda, bool)
            or not (0.0 <= mmr_lambda <= 1.0)
        ):
            # Treat out-of-range MMR lambda as default (search degrades but
            # still produces results) — Plan §Phase 3.1 doesn't list this as
            # an HTTP error case. A bool is rejected the same way: silently
            # coercing ``true`` to 1.0 would disable MMR diversity entirely.
            mmr_lambda = self._settings.default_mmr_lambda

        limit = payload.get("limit", self._settings.default_limit)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise SearchValidationError("INVALID_LIMIT", "limit must be a positive int")
        limit = min(limit, self._settings.max_limit)

        edge_types = payload.get("edge_types", []) or []
        if not isinstance(edge_types, list) or not all(isinstance(e, str) for e in edge_types):
            raise SearchValidationError(
                "INVALID_EDGE_TYPES", "edge_types must be a list of strings"
            )

        return {
            "seeds": seeds,
            "max_hops": max_hops,
            "min_cosine": float(min_cosine),
            "mmr_lambda": float(mmr_lambda),
            "limit": limit,
            "edge_types": edge_types,
            "include_episodes": bool(payload.get("include_episodes", False)),
        }

    # ─── pipeline ──────────────────────────────────────────────────────

    async def search(self, payload: Dict[str, Any]) -> SearchResponse:
        """Run the BFS+cosine+MMR pipeline. End-to-end timed."""
        start_ts = time.monotonic()
        req = self.validate_request(payload)

        if self._redis is None:
            await self.start()
        if self._redis is None:
            raise SearchBackendError(
                "FALKOR_DOWN",
                "FalkorDB unreachable — check daemon logs for circuit state",
            )

        warnings: List[str] = []

        # Step 1 — seed existence probe (fast path for 404).
        existing = await self._probe_seed_existence(req["seeds"])
        if not existing:
            raise SearchBackendError("SEED_NOT_FOUND", "no seed_entities exist in the KG")

        # Step 2 — Cypher BFS with timeout.
        try:
            rows = await asyncio.wait_for(
                self._cypher_bfs(req),
                timeout=self._settings.traversal_timeout_sec,
            )
        except asyncio.TimeoutError:
            raise SearchBackendError(
                "TRAVERSAL_TIMEOUT",
                f"BFS exceeded {self._settings.traversal_timeout_sec}s budget",
            )

        if len(rows) >= self._settings.max_facts_per_query:
            warnings.append("fact_count_capped: results may be truncated")

        # Step 3 — extract FactRows (already extracted by Cypher RETURN clause).
        fact_rows: List[FactRow] = list(rows)

        # Step 4 — cosine via EmbeddingClient (batched per fact).
        if fact_rows:
            await self._attach_cosine(fact_rows, payload.get("query"))

        # Step 5 — filter: cosine < min_cosine OR invalid_at != None.
        filtered = [
            r
            for r in fact_rows
            if r.invalid_at is None and r.cosine is not None and r.cosine >= req["min_cosine"]
        ]

        # Step 6 — MMR rerank. Falls through to cosine-only ordering if
        # mmr_lambda == 0 (pure diversity) or == 1 (pure relevance).
        ranked = self._mmr_rerank(filtered, req["mmr_lambda"], req["limit"])

        elapsed_ms = int((time.monotonic() - start_ts) * 1000)
        return SearchResponse(
            results=ranked[: req["limit"]],
            count=len(ranked[: req["limit"]]),
            elapsed_ms=elapsed_ms,
            degraded=bool(warnings) or any(r.cosine is None for r in fact_rows),
            warnings=warnings,
        )

    # ─── backend steps ─────────────────────────────────────────────────

    async def _probe_seed_existence(self, seeds: List[str]) -> List[str]:
        """Return subset of seeds that exist in the KG.

        FalkorDB params are baked into the Cypher query as a ``CYPHER``
        header. We build that header here for the single ``$name`` param.
        """
        existing: List[str] = []
        for name in seeds:
            # f-string with backticks around the param name (per falkordb client).
            cypher = f"CYPHER `name`={_cypher_param(name)} {_CYPHER_EXISTS}"
            try:
                result = await self._redis.execute_command(  # type: ignore[union-attr]
                    "GRAPH.QUERY",
                    "graphiti_memory",
                    cypher,
                    "--compact",
                )
                # Falkor returns [headers, rows, stats]; first data row = [n.name]
                if isinstance(result, list) and len(result) >= 2 and result[1]:
                    existing.append(name)
            except Exception as e:
                # Do NOT swallow this. An empty ``existing`` is read by the
                # caller as "seed absent" -> HTTP 404 SEED_NOT_FOUND, which
                # tells the client to fix a query that is actually fine and
                # blames their seed name for a dead database. Re-raise through
                # the same classifier _cypher_bfs uses, so a transport failure
                # reads FALKOR_DOWN (503) and a query-text defect reads
                # INVALID_CYPHER (400, no circuit-breaker impact) -- see
                # _classify_cypher_error for why the split matters.
                raise _classify_cypher_error(e) from e
        return existing

    async def _cypher_bfs(self, req: Dict[str, Any]) -> List[FactRow]:
        """Execute Cypher BFS and parse rows into FactRow.

        Schema-aware extraction: KG v1 stores "facts" as the target node's
        ``summary`` property (multi-line text). For each hop, we synthesize
        a FactRow from the neighbor node's summary + the relationship type
        in the path.
        """
        seed_literals = ", ".join(f"'{_escape_cypher_string(name)}'" for name in req["seeds"])
        cypher = (
            _CYPHER_BFS.replace("{seed_literals}", seed_literals)
            .replace("{max_hops}", str(req["max_hops"]))
            .replace("{cap}", str(int(self._settings.max_facts_per_query)))
        )

        try:
            result = await self._redis.execute_command(  # type: ignore[union-attr]
                "GRAPH.QUERY",
                "graphiti_memory",
                cypher,
                "--compact",
            )
        except Exception as e:
            raise _classify_cypher_error(e)

        rows = _parse_falkor_rows(result)
        out: List[FactRow] = []
        for row in rows:
            try:
                # Query returns (edge_type, hop, path_nodes, edge_fact,
                # seed_in_path, target_in_path) in 6 flat columns, read
                # positionally. Since 2026-10-04 the 4th column is the
                # relationship's own fact, one row per rel -- previously it
                # was the SEED node's summary repeated on every row.
                edge_type, hop, path_nodes, edge_fact, seed_in_path, target_in_path = row[:6]
                if not isinstance(edge_fact, str) or not edge_fact.strip():
                    continue
                first_line = next(
                    (ln.strip() for ln in edge_fact.splitlines() if ln.strip()),
                    "",
                )
                if not first_line:
                    continue
                hop_count = int(hop) if hop is not None else 1
                out.append(
                    FactRow(
                        fact=first_line,
                        source_entity=str(seed_in_path)
                        if seed_in_path
                        else (path_nodes[0] if path_nodes else ""),
                        target_entity=str(target_in_path) if target_in_path else "",
                        edge_type=str(edge_type) if edge_type else "RELATES_TO",
                        hop_distance=hop_count,
                        path=list(path_nodes) if path_nodes else [],
                        invalid_at=None,
                    )
                )
            except Exception as e:
                logger.debug("malformed bfs row %r: %s", row, e)
        return out

    async def _attach_cosine(self, rows: List[FactRow], query: Optional[str]) -> None:
        """Embed each fact via EmbeddingClient; compute cosine to query.

        If query is not provided, we use the first seed as the implicit
        query (callers usually pass it via the request payload — the
        handler sets ``payload["query"] = payload.get("seed_entities", [""])[0]``).
        """
        if not query:
            query = rows[0].fact if rows else ""
        texts = [r.fact for r in rows] + [query]
        embeds = await self._embed.embed(texts)
        if not embeds or any(e.embedding is None for e in embeds):
            # Embedding failed → leave cosine=None; pipeline flags degraded.
            return
        query_vec = embeds[-1].embedding
        for idx, row in enumerate(rows):
            row.cosine = (
                _cosine(query_vec, embeds[idx].embedding) if embeds[idx].embedding else None
            )

    def _mmr_rerank(self, rows: Sequence[FactRow], lam: float, top_k: int) -> List[FactRow]:
        """Maximal Marginal Relevance rerank.

        Standard MMR:
          score(d) = lam * sim(d, query) - (1 - lam) * max(sim(d, d_selected))

        Rows without a cosine (embedding failed) sort to the bottom — we
        do NOT drop them, since the caller may still want raw BFS hits.
        """
        if not rows:
            return []
        if lam <= 0.0:
            # Pure diversity — order by hop_distance ascending, then arbitrary.
            return sorted(rows, key=lambda r: (r.hop_distance, -float(r.cosine or 0.0)))
        if lam >= 1.0:
            # Pure relevance — cosine-only ordering.
            return sorted(rows, key=lambda r: -float(r.cosine or 0.0))

        # Diversity penalty: (1 - lam) × hop_distance × SCALE × (gap + 0.2).
        # SCALE=2.0 calibrated so the canonical test in test_search_graph.py
        # shows lam=0.7 reorders rows: high_far (cos 0.95, hop 3) loses to
        # mid_near (cos 0.60, hop 1) because the diversity penalty (~0.36)
        # outweighs the 0.35 cosine gap. lam=1 → penalty=0 (pure relevance).
        # lam=0 → hop_distance dominates (pure diversity).
        max_cos = max((float(r.cosine or 0.0) for r in rows), default=0.0)

        def _key(r: FactRow):
            cosine = float(r.cosine or 0.0)
            gap = max_cos - cosine + 0.2
            diversity_penalty = (1.0 - lam) * 2.0 * float(r.hop_distance) * gap
            return -(cosine - diversity_penalty)

        return sorted(rows, key=_key)[:top_k]


# ─── pure helpers (testable in isolation) ──────────────────────────────


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Returns 0.0 on zero vector or length mismatch."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _params_to_json(params: Dict[str, Any]) -> str:
    """Serialize dict to compact JSON string for Falkor params positional arg."""
    import json

    return json.dumps(params, ensure_ascii=False, separators=(",", ":"))


def _parse_falkor_rows(result: Any) -> List[List[Any]]:
    """Falkor returns ``[headers, rows, stats]`` in row-of-rows encoding.

    With ``--compact`` (which we always use), each cell is wrapped in a
    ``[type_code, value]`` tuple:
      - ``[1, "column_name"]`` for headers (type 1 = string)
      - ``[2, "string_value"]`` for string cells
      - ``[3, integer_value]`` for int cells
      - ``[4, float_value]`` for float cells
      - ``[6, [...]]`` for list cells (paths, embeddings, etc.)
      - ``[5, null]`` / ``[0, ...]`` for null/missing

    This helper unwraps those tuples so callers receive the raw Python
    values directly: strings, ints, floats, lists, None.

    Returns:
        List of rows, each row a list of column values. Header NOT
        included (caller does not need to filter it).
    """
    if not isinstance(result, list) or len(result) < 2:
        return []

    raw_rows = result[1] or []
    if not isinstance(raw_rows, list):
        return []

    def _unwrap(cell: Any) -> Any:
        """Unwrap ``[type_code, value]`` compact cell. Returns value as-is
        if not a 2-element list.
        """
        if isinstance(cell, list) and len(cell) == 2 and isinstance(cell[0], int):
            return cell[1]
        return cell

    out: List[List[Any]] = []
    for row in raw_rows:
        if not isinstance(row, list):
            out.append([_unwrap(row)])
            continue
        out.append([_unwrap(c) for c in row])
    return out


def _classify_cypher_error(exc: Exception) -> SearchBackendError:
    r"""Decide whether a Cypher failure is a client-input error or a dead engine.

    A query built from *our own* validated input should never be a syntax
    error. When one happens it means the generated text is malformed — a
    server-side defect, not a FalkorDB outage. Reporting it as FALKOR_DOWN
    (503) is actively harmful: it tells monitoring the database is down and
    can trip the daemon's circuit-breaker / restart policy over a bug that
    has nothing to do with connectivity.

    So: a syntax/parse complaint becomes INVALID_CYPHER (400, client-visible,
    no circuit-breaker impact); genuine connection/transport failures stay
    FALKOR_DOWN (503).

    Split is conservative — anything that does not clearly look like a query
    parse problem is still treated as the backend being down, so a real
    outage is never masked by this change.
    """
    msg = str(exc).lower()
    parse_markers = (
        "invalid input",
        "syntax error",
        "errormessage",
        "expected",
        "parse",
    )
    if any(marker in msg for marker in parse_markers):
        return SearchBackendError(
            "INVALID_CYPHER",
            f"generated Cypher query was rejected by FalkorDB (not an outage): {exc}",
        )
    return SearchBackendError("FALKOR_DOWN", f"cypher failed: {exc}")


def _escape_cypher_string(value: str) -> str:
    r"""Escape a Python string for a single-quoted Cypher string literal.

    FalkorDB (v8.9.x, verified 2026-10-01 against graphiti_memory) uses
    **backslash** escaping inside ``'...'`` — NOT the SQL-style quote-doubling
    (``''``) that openCypher/Neo4j docs imply. Escape the backslash FIRST,
    then the single quote, otherwise a value ending in a backslash would
    swallow the closing quote.

    Measured round-trip over 12 payloads (live :6379, ``RETURN <literal>``):
      backslash-first -> 12/12 exact, 0 corrupted, 0 errors
      quote-doubling  ->  5 exact,  2 corrupted, 5 parse errors

    Quote-doubling fails two ways, both verified:
      1. PARSE ERROR — ``'O'Brien'`` makes the lexer see ``'O'`` followed by
         a bare identifier, so the whole query is rejected.
      2. SILENT CORRUPTION — ``'a\\b'`` is read as escape sequence ``\\b``
         and returns ``'a\x08'`` (backspace). No error, wrong data.

    Backslash-first round-trips quotes, backslashes, tabs, newlines,
    semicolons and braces exactly.

    Parameter binding (``... PARAMS n k v``) is NOT available as an
    alternative here — this FalkorDB rejects it with
    ``Invalid input 'P'``. Literal escaping is the only option.
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _cypher_param(value: Any) -> str:
    """Format a Python value as a FalkorDB CYPHER header literal.

    Mirrors falkordb's stringify_param_value: strings get single-quoted
    with backslash escaping (see ``_escape_cypher_string`` — this engine
    does NOT accept SQL-style ``''`` doubling); numbers / bools / None get
    their repr; lists/objects NOT supported here (v1 always inlines literal
    lists directly in the query).
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = _escape_cypher_string(value)
        return f"'{escaped}'"
    # Fallback: string repr with quotes (best-effort).
    return f"'{value!s}'"
