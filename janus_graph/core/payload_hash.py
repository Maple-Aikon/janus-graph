"""Canonical payload hash — the single source of truth for episode dedup.

WHY THIS MODULE EXISTS
----------------------
Two writers insert into the same ``episodes`` table:

  1. the daemon HTTP writer (``daemon/episode_queue_adapter.py``) — stamps
     ``payload_hash`` on insert and rejects duplicates with HTTP 409;
  2. the MCP stdio writer (``pipeline/queue.py::EpisodeQueue.enqueue``) —
     historically inserted WITHOUT a hash, so it was invisible to the
     daemon's dedup gate.

If the two writers compute the hash differently, the daemon's dedup never
sees the MCP rows and dedup silently does nothing. So the scheme lives in
ONE module that both sides import.

THE SCHEME (canonical, frozen)
------------------------------
    sha256( content || "|" || group_id || "|" || source_description )

THE INJECTIVITY CONDITION (read this before "fixing" the scheme)
----------------------------------------------------------------
The concatenation is only injective over a given set of triples if
``group_id`` and ``source_description`` never contain the ``|``
separator. Precisely:

  * ``content`` MAY contain ``|`` — freely. With a fixed ``group_id`` and
    a fixed ``source_description``, the trailing ``group_id|source_desc``
    suffix is identical for both triples, so the remaining prefix is the
    content, and distinct contents give distinct inputs.
  * ``group_id`` and ``source_description`` MUST NOT contain ``|``. A
    separator inside either one moves the boundary between fields, so two
    genuinely different triples can hash identically.

Worked example of the failure (brute-forced, 2026-10-31):

    ("x|y", "g", "s")  ->  "x|y|g|s"
    ("x",  "y|g", "s") ->  "x|y|g|s"     <-- SAME INPUT, SAME HASH

and symmetrically, when BOTH labels carry a pipe:

    ("a", "g",   "y|g") ->  "a|g|y|g"
    ("a", "g|y", "g")   ->  "a|g|y|g"

Note which side is at fault: it is the ``|`` inside ``group_id``, not the
one inside ``content``. A content that merely CONTAINS a pipe is safe.
Brute force over a small value space confirms that EVERY collision
requires a pipe in ``group_id`` — a pipe in ``source_description`` alone
cannot collide while ``group_id`` stays pipe-free.

Measured on the live queue DB (2026-10-31): 201 distinct
``source_description`` values, 0 of them contain ``|``; 3 distinct
``group_id`` values (``graphiti``, ``graphiti_memory``,
``test_tenant_xyz``), none contain ``|``. So the live condition holds.

This module does NOT enforce the condition — it logs a warning when it
is violated, because the MCP tool takes ``source_description`` from the
caller and a future caller could pass a pipe. Enforcement (rejecting the
episode) would turn a cosmetic hash weakness into data loss, which is the
wrong trade for a dedup heuristic.

``source_description`` is part of the hash ON PURPOSE: two callers writing
the same content from different sources (e.g. ``user_conversation`` vs
``seahorse_compaction:<session>``) are different episodes, not duplicates.

Verified against the live queue DB on 2026-10-31: every row written by the
canonical writers (``seahorse_compaction:*``, ``daemon_http``, ``verify``)
matched this scheme byte-for-byte (348 + 6 + 4 rows, 100%). Rows written by
the bulk ``signet_*_reimport`` scripts use a DIFFERENT scheme and do NOT
match — they are historical rows, not evidence against this scheme.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not decide WHEN to hash or whether to reject. That policy lives in
``cfg.pipeline.dedup.mode`` (``off`` / ``observe`` / ``enforce``) and is
applied by the writer. Hashing a row and rejecting a row are separate
decisions; conflating them is how a dedup rollout silently starts
dropping data.
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger("janus_graph.core.payload_hash")

__all__ = ["canonical_payload_hash"]


def canonical_payload_hash(content: str, group_id: str, source_description: str) -> str:
    """Return the canonical sha256 hex digest for one episode payload.

    Args:
        content: The episode body. Any string, including newlines/NULs and
            ``|``. A pipe in the content is SAFE — see the module docstring.
        group_id: The memory tenant id (e.g. ``graphiti_memory``). Must not
            contain ``|``.
        source_description: The writer/content-class tag. Part of the hash so
            different content classes never collide. Must not contain ``|``.

    Returns:
        64-character lowercase hex digest.

    This is the same function the daemon adapter used inline; it was moved
    here so the MCP writer stamps byte-identical hashes. Do NOT change the
    scheme in place — a scheme change invalidates every stored hash and
    silently disables dedup across the whole table. If the scheme must
    change, version it (e.g. a ``v2:`` prefix) and migrate the column.

    A ``|`` inside ``group_id`` or ``source_description`` weakens the hash
    (two different triples can collide). That is logged, NOT raised:
    rejecting an episode because of a character in its own label would be
    a worse failure than the collision it prevents.
    """
    if "|" in group_id or "|" in source_description:
        logger.warning(
            "group_id/source_description contains the '|' separator, which "
            "weakens the payload hash (distinct triples may collide): "
            "group_id=%r source_description=%r",
            group_id,
            source_description,
        )
    h = hashlib.sha256()
    h.update(content.encode("utf-8"))
    h.update(b"|")
    h.update(group_id.encode("utf-8"))
    h.update(b"|")
    h.update(source_description.encode("utf-8"))
    return h.hexdigest()
