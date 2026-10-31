"""1b — enqueue-level dedup on the MCP writer path.

Context
-------
Two writers insert into the same ``episodes`` table:

  * the daemon HTTP writer — stamps ``payload_hash`` and rejects duplicates
    with HTTP 409;
  * the MCP stdio writer (``EpisodeQueue.enqueue``) — historically inserted
    with NO hash, so the daemon's gate could never see MCP rows. Dedup was
    effectively a one-writer affair.

1b closes that gap, behind ``cfg.pipeline.dedup.mode`` with three modes:
``off`` (default), ``observe``, ``enforce``.

What these tests pin
--------------------
  1. DEFAULT IS OFF — with no configuration, enqueue behaves exactly as
     before: no hash stored, duplicates accepted. This is the safety
     property: shipping a dedup gate that silently drops episodes is worse
     than shipping no gate.
  2. OBSERVE — hash IS stored, duplicates ARE detected and counted, but
     the episode is STILL enqueued. Nothing is ever rejected.
  3. ENFORCE — a duplicate is rejected and the caller learns the id of the
     row that already holds the payload.
  4. HASH SCHEME — the MCP writer's hash is byte-identical to the daemon
     writer's. If these diverge, the daemon's gate silently stops seeing
     MCP rows and dedup does nothing while looking enabled.
  5. ACTIVE-ONLY — rows already ``done``/``failed`` do NOT block a new
     enqueue. Re-queueing after a permanent failure is a legitimate retry;
     blocking it would strand the row forever.
  6. UNKNOWN MODE — falls back to ``off``, never to ``enforce``. A typo in
     a config value must not start dropping episodes.
  7. COUNTERS — ``get_dedup_stats()`` reports what the observe rollout
     needs to be judged from a number rather than a feeling.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from janus_graph.core.payload_hash import canonical_payload_hash
from janus_graph.pipeline.queue import (
    DuplicateEpisodeError,
    EpisodeQueue,
    _resolve_dedup_mode,
)


def _tmp_queue() -> EpisodeQueue:
    db_path = Path(tempfile.mkdtemp(prefix="jg_dedup_")) / "episodes.db"
    return EpisodeQueue(str(db_path))


def _row(q: EpisodeQueue, episode_id: str):
    with sqlite3.connect(str(q.db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()


def _count(q: EpisodeQueue) -> int:
    with sqlite3.connect(str(q.db_path)) as conn:
        return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]


# ── 1. default is off ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_default_off_stores_no_hash_and_accepts_duplicates():
    """With no config, behaviour is byte-for-byte the pre-1b behaviour."""
    q = _tmp_queue()

    ep1 = await q.enqueue("same content", group_id="g1", source_description="user_conversation")
    ep2 = await q.enqueue("same content", group_id="g1", source_description="user_conversation")

    assert ep1 != ep2, "duplicate must still be accepted in default mode"
    assert _count(q) == 2

    for ep_id in (ep1, ep2):
        row = _row(q, ep_id)
        assert row["payload_hash"] is None, "default mode must not stamp a hash"
        assert row["claimed_by"] is None


@pytest.mark.asyncio
async def test_default_off_resolves_to_off():
    assert _resolve_dedup_mode() == "off"


# ── 2. observe mode ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_observe_stores_hash_but_still_enqueues_duplicate():
    """Observe detects and counts, but NEVER rejects."""
    q = _tmp_queue()

    ep1 = await q.enqueue(
        "dup me", group_id="g1", source_description="user_conversation", dedup_mode="observe"
    )
    ep2 = await q.enqueue(
        "dup me", group_id="g1", source_description="user_conversation", dedup_mode="observe"
    )

    assert ep1 != ep2, "observe mode must NOT reject the duplicate"
    assert _count(q) == 2, "observe mode must still enqueue the duplicate"

    row = _row(q, ep1)
    assert row["payload_hash"] is not None, "observe mode must store the hash"
    assert row["payload_hash"] == canonical_payload_hash("dup me", "g1", "user_conversation")
    assert row["claimed_by"] is not None and row["claimed_by"].startswith("mcp:")

    stats = q.get_dedup_stats()
    assert stats["observed_hits"] == 1, "observe mode must count the duplicate"
    assert stats["mode"] == "off", "stats report the NEXT enqueue's configured mode"


@pytest.mark.asyncio
async def test_observe_counts_only_real_duplicates():
    """Distinct content is not a duplicate."""
    q = _tmp_queue()

    await q.enqueue("a", group_id="g", source_description="s", dedup_mode="observe")
    await q.enqueue("b", group_id="g", source_description="s", dedup_mode="observe")
    await q.enqueue("a", group_id="g", source_description="s", dedup_mode="observe")

    assert q.get_dedup_stats()["observed_hits"] == 1


@pytest.mark.asyncio
async def test_observe_does_not_flag_different_source_description():
    """Same content, different content-class ⇒ NOT a duplicate.

    This is why ``source_description`` is part of the hash: a
    ``conversation_summary`` and a ``seahorse_compaction`` carrying the
    same body are different episodes.
    """
    q = _tmp_queue()

    await q.enqueue(
        "same", group_id="g", source_description="user_conversation", dedup_mode="observe"
    )
    await q.enqueue(
        "same", group_id="g", source_description="seahorse_compaction:x", dedup_mode="observe"
    )

    assert q.get_dedup_stats()["observed_hits"] == 0
    assert _count(q) == 2


@pytest.mark.asyncio
async def test_observe_does_not_flag_different_group_id():
    q = _tmp_queue()

    await q.enqueue("same", group_id="g1", source_description="s", dedup_mode="observe")
    await q.enqueue("same", group_id="g2", source_description="s", dedup_mode="observe")

    assert q.get_dedup_stats()["observed_hits"] == 0


# ── 3. enforce mode ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_enforce_rejects_duplicate_and_reports_existing_id():
    q = _tmp_queue()

    ep1 = await q.enqueue(
        "dup me", group_id="g1", source_description="user_conversation", dedup_mode="enforce"
    )

    with pytest.raises(DuplicateEpisodeError) as excinfo:
        await q.enqueue(
            "dup me", group_id="g1", source_description="user_conversation", dedup_mode="enforce"
        )

    assert excinfo.value.existing_episode_id == ep1
    assert _count(q) == 1, "enforce must NOT insert the duplicate"


@pytest.mark.asyncio
async def test_enforce_allows_distinct_content():
    q = _tmp_queue()

    ep1 = await q.enqueue("one", group_id="g", source_description="s", dedup_mode="enforce")
    ep2 = await q.enqueue("two", group_id="g", source_description="s", dedup_mode="enforce")

    assert ep1 != ep2
    assert _count(q) == 2


# ── 4. hash scheme parity with the daemon writer ───────────────────────


def test_mcp_hash_matches_daemon_hash():
    """The two writers MUST agree byte-for-byte.

    If they diverge, the daemon's dedup gate cannot see the MCP writer's
    rows — dedup silently stops working while the config still says it is
    on. This is the exact failure mode 1b exists to prevent.
    """
    from janus_graph.daemon.episode_queue_adapter import _payload_hash

    cases = [
        ("hello", "graphiti_memory", "user_conversation"),
        ("multi\nline\nbody", "g", "s"),
        ("", "g", "s"),
        ("unicode: xin chào 🌊", "g", "s"),
        ("pipe|inside|content", "g", "s"),
    ]
    for content, gid, sd in cases:
        assert canonical_payload_hash(content, gid, sd) == _payload_hash(content, gid, sd)


def test_hash_is_injective_when_labels_are_pipe_free():
    """The REAL invariant: group_id and source_description pipe-free ⇒ injective.

    A pipe in the CONTENT is safe — with a fixed group_id and
    source_description, the trailing suffix is identical for both triples
    and the remaining prefix is the content.
    """
    a = canonical_payload_hash("x|y", "g", "s")
    b = canonical_payload_hash("x", "g", "s")
    assert a != b, "a pipe in the content must not collide"

    # Different content, same labels — all distinct.
    hashes = {
        canonical_payload_hash(c, "g", "s")
        for c in ("a", "b", "a|b", "|", "", "x|y", "multi\nline")
    }
    assert len(hashes) == 7, "distinct contents must hash distinctly"

    # Different source_description (pipe-free), same content and group.
    assert canonical_payload_hash("x", "g", "s1") != canonical_payload_hash("x", "g", "s2")

    # Different group_id (pipe-free), same content and source.
    assert canonical_payload_hash("x", "g1", "s") != canonical_payload_hash("x", "g2", "s")


def test_known_collision_when_group_id_contains_pipe():
    """Pin the KNOWN weakness instead of pretending it does not exist.

    A pipe inside ``group_id`` (or ``source_description``) moves the
    field boundary, so two different triples can hash identically:

        ("x|y", "g",   "s") -> "x|y|g|s"
        ("x",   "y|g", "s") -> "x|y|g|s"

    Brute-forced over a small value space on 2026-10-31: EVERY collision
    requires a pipe in ``group_id`` (or in ``source_description`` paired
    with a pipe in ``group_id``). With both labels pipe-free the scheme
    is injective, which is what
    ``test_hash_is_injective_when_labels_are_pipe_free`` pins.

    The module logs a warning for the weak case rather than rejecting the
    episode. A test asserting the collision does NOT happen would
    document a falsehood — this one pins reality so a future scheme
    change is caught.
    """
    a = canonical_payload_hash("x|y", "g", "s")
    b = canonical_payload_hash("x", "y|g", "s")
    assert a == b, "documented collision: a pipe in group_id shifts the boundary"

    # A pipe in source_description collides only when group_id carries one
    # too (the boundary has to move somewhere):
    #   ("a", "g",   "y|g") -> "a|g|y|g"
    #   ("a", "g|y", "g")   -> "a|g|y|g"
    c = canonical_payload_hash("a", "g", "y|g")
    d = canonical_payload_hash("a", "g|y", "g")
    assert c == d

    # But with a pipe-free group_id, a pipe in source_description is safe.
    e = canonical_payload_hash("a", "g", "y|g")
    f = canonical_payload_hash("a", "g", "y|g")
    assert e == f  # same triple, trivially equal
    assert canonical_payload_hash("a", "g", "y|g") != canonical_payload_hash("a", "g", "other")


def test_pipe_in_label_logs_warning_but_still_hashes(caplog):
    """A weak hash is a warning, never an exception.

    Rejecting an episode because of a character in its own label would be
    a worse failure than the collision it prevents.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="janus_graph.core.payload_hash"):
        digest = canonical_payload_hash("x", "y|g", "s")

    assert len(digest) == 64
    assert any("separator" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_enforce_hash_matches_canonical_for_stored_row():
    q = _tmp_queue()

    ep_id = await q.enqueue(
        "content here", group_id="g1", source_description="user_conversation", dedup_mode="enforce"
    )

    row = _row(q, ep_id)
    assert row["payload_hash"] == canonical_payload_hash("content here", "g1", "user_conversation")


# ── 5. active-status semantics ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_done_row_does_not_block_new_enqueue():
    """A finished row must not block: re-queueing is a legitimate retry."""
    q = _tmp_queue()

    ep1 = await q.enqueue("retry me", group_id="g", source_description="s", dedup_mode="enforce")
    await q.mark_done(ep1)

    ep2 = await q.enqueue("retry me", group_id="g", source_description="s", dedup_mode="enforce")

    assert ep2 != ep1
    assert _count(q) == 2


@pytest.mark.asyncio
async def test_failed_row_does_not_block_new_enqueue():
    q = _tmp_queue()

    ep1 = await q.enqueue("retry me", group_id="g", source_description="s", dedup_mode="enforce")
    await q.mark_failed(ep1, "boom")

    ep2 = await q.enqueue("retry me", group_id="g", source_description="s", dedup_mode="enforce")

    assert ep2 != ep1


@pytest.mark.asyncio
async def test_processing_row_does_block():
    """A row being processed right now IS a live duplicate."""
    q = _tmp_queue()

    ep1 = await q.enqueue("in flight", group_id="g", source_description="s", dedup_mode="enforce")
    await q.claim_next_batch(limit=1)
    assert _row(q, ep1)["status"] == "processing"

    with pytest.raises(DuplicateEpisodeError):
        await q.enqueue("in flight", group_id="g", source_description="s", dedup_mode="enforce")


# ── 6. unknown mode handling ───────────────────────────────────────────


def test_config_layer_folds_unknown_mode_instead_of_raising():
    """A typo'd flag must NOT brick the MCP server.

    ``JanusSettings`` is built at MCP server startup. With a pydantic
    ``Literal``, a typo in the operator's YAML would raise there and take
    the whole server down — no memory tools at all. A bad flag value
    must degrade to the safe mode, not to a dead process.
    """
    import os

    from janus_graph.config import JanusSettings

    os.environ["JANUS_PIPELINE__DEDUP__MODE"] = "enforece"
    try:
        s = JanusSettings()
        assert s.pipeline.dedup.mode == "off", "unknown mode must fold to off"
    finally:
        os.environ.pop("JANUS_PIPELINE__DEDUP__MODE", None)


def test_config_layer_accepts_all_three_valid_modes():
    import os

    from janus_graph.config import JanusSettings

    for mode in ("off", "observe", "enforce"):
        os.environ["JANUS_PIPELINE__DEDUP__MODE"] = mode
        try:
            s = JanusSettings()
            assert s.pipeline.dedup.mode == mode
        finally:
            os.environ.pop("JANUS_PIPELINE__DEDUP__MODE", None)


def test_config_layer_normalizes_case_and_whitespace():
    import os

    from janus_graph.config import JanusSettings

    os.environ["JANUS_PIPELINE__DEDUP__MODE"] = "  Enforce  "
    try:
        s = JanusSettings()
        assert s.pipeline.dedup.mode == "enforce"
    finally:
        os.environ.pop("JANUS_PIPELINE__DEDUP__MODE", None)


def test_unknown_mode_falls_back_to_off_not_enforce():
    """A typo must never start DROPPING episodes.

    Falling back to ``enforce`` on an unrecognised value would turn a
    config typo into silent data loss. Falling back to ``off`` only means
    the gate is not applied — a safe, visible no-op.
    """
    assert _resolve_dedup_mode("enforece") == "off"
    assert _resolve_dedup_mode("") == "off"
    assert _resolve_dedup_mode("ENFORCE") == "enforce", "case must not matter"
    assert _resolve_dedup_mode("  observe  ") == "observe", "whitespace must not matter"


@pytest.mark.asyncio
async def test_unknown_mode_does_not_reject():
    q = _tmp_queue()

    ep1 = await q.enqueue("x", group_id="g", source_description="s", dedup_mode="enforece")
    ep2 = await q.enqueue("x", group_id="g", source_description="s", dedup_mode="enforece")

    assert ep1 != ep2
    assert _count(q) == 2


# ── 7. counters / stats ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dedup_stats_shape():
    q = _tmp_queue()

    stats = q.get_dedup_stats()
    assert set(stats) == {"mode", "observed_hits", "hashed_rows", "active_duplicates"}
    assert stats["observed_hits"] == 0
    assert stats["hashed_rows"] == 0
    assert stats["active_duplicates"] == 0

    await q.enqueue("a", group_id="g", source_description="s", dedup_mode="observe")
    stats = q.get_dedup_stats()
    assert stats["hashed_rows"] == 1
    assert stats["active_duplicates"] == 0


@pytest.mark.asyncio
async def test_dedup_stats_counts_active_duplicate_groups():
    """Two rows sharing a hash while active ⇒ 1 duplicate group."""
    q = _tmp_queue()

    await q.enqueue("same", group_id="g", source_description="s", dedup_mode="observe")
    await q.enqueue("same", group_id="g", source_description="s", dedup_mode="observe")

    assert q.get_dedup_stats()["active_duplicates"] == 1


def test_get_stats_not_polluted_by_dedup_keys():
    """``get_stats()`` is summed by callers — it must keep status keys only.

    ``dream.py`` computes ``done + queued`` from it; adding a non-status
    key would silently corrupt that arithmetic.
    """
    q = _tmp_queue()
    stats = q.get_stats()
    assert "observed_hits" not in stats
    assert "mode" not in stats
    assert all(k in {"queued", "processing", "done", "failed", "aborted", "dlq"} for k in stats)


# ── 8. config lookup is cached (hot path) ──────────────────────────────


def test_config_resolved_mode_is_cached(monkeypatch):
    """``load_config()`` re-parses the YAML on every call (5.4 ms measured).

    Resolving the mode per-enqueue would add that cost to every MCP
    ``add_episode`` call — a real regression for a gate whose default is
    "off". The resolved mode is cached process-wide.
    """
    from janus_graph.pipeline import queue as q_mod

    calls = []

    def _fake_load_config():
        calls.append(1)
        from janus_graph.config import JanusSettings

        return JanusSettings()

    q_mod.reset_dedup_mode_cache()
    try:
        monkeypatch.setattr("janus_graph.config.load_config", _fake_load_config)

        q_mod._resolve_dedup_mode()
        q_mod._resolve_dedup_mode()
        q_mod._resolve_dedup_mode()

        assert len(calls) == 1, f"config must be read once, got {len(calls)} reads"
    finally:
        q_mod.reset_dedup_mode_cache()


def test_reset_dedup_mode_cache_forces_reread(monkeypatch):
    """The cache must be resettable — otherwise a config change never lands."""
    from janus_graph.pipeline import queue as q_mod

    calls = []

    def _fake_load_config():
        calls.append(1)
        from janus_graph.config import JanusSettings

        return JanusSettings()

    q_mod.reset_dedup_mode_cache()
    try:
        monkeypatch.setattr("janus_graph.config.load_config", _fake_load_config)
        q_mod._resolve_dedup_mode()
        q_mod.reset_dedup_mode_cache()
        q_mod._resolve_dedup_mode()
        assert len(calls) == 2
    finally:
        q_mod.reset_dedup_mode_cache()


def test_explicit_mode_bypasses_config_entirely(monkeypatch):
    """An explicit mode must NOT touch the config file at all."""
    from janus_graph.pipeline import queue as q_mod

    def _boom():
        raise AssertionError("config must not be read when the mode is explicit")

    q_mod.reset_dedup_mode_cache()
    try:
        monkeypatch.setattr("janus_graph.config.load_config", _boom)
        assert q_mod._resolve_dedup_mode("observe") == "observe"
        assert q_mod._resolve_dedup_mode("enforce") == "enforce"
    finally:
        q_mod.reset_dedup_mode_cache()


# ── 9. schema ──────────────────────────────────────────────────────────


def test_columns_exist_on_fresh_db():
    """A queue created against a fresh DB (never touched by the daemon)
    must still have the dedup columns, or the MCP INSERT would fail."""
    q = _tmp_queue()

    with sqlite3.connect(str(q.db_path)) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(episodes)")}
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}

    assert "payload_hash" in cols
    assert "claimed_by" in cols
    assert "idx_episodes_payload_hash" in indexes
