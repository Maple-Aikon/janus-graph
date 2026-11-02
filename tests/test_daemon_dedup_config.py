"""The daemon HTTP writer must honour ``cfg.pipeline.dedup.mode``.

WHY THIS FILE EXISTS
---------------------
The daemon HTTP writer (``daemon/episode_queue_adapter.py``) used to
HARDCOD ``DEDUP_GATE_ENFORCE``: it stamped ``dedup_gate='enforce'`` on
every row and rejected every duplicate, no matter what
``cfg.pipeline.dedup.mode`` said. The MCP stdio writer, the PicoClaw
hook and the daemon were three writers into one table with THREE
different policies — and the one an operator could not turn off was the
daemon.

These tests pin the FIXED behaviour: the daemon resolves its mode from
the same config the other writers read, and each of the three modes
behaves exactly as the MCP stdio writer's mode behaves.

The contract being pinned (identical to the MCP writer):

  off      — no lookup, no ledger row, no legacy hash stamp, duplicate
             ACCEPTED. Byte-for-byte the pre-dedup behaviour.
  observe  — duplicate DETECTED and RECORDED (``action='detected'``),
             but ACCEPTED. Observe must never reject.
  enforce  — duplicate DETECTED, RECORDED (``action='rejected'``) and
             REJECTED. The row is stamped ``dedup_gate='enforce'`` so the
             UNIQUE partial index covers it.

And the safety rule: an UNRECOGNISED mode folds to ``off``, never to
``enforce``. A typo in the operator's config must not start dropping
episodes.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3

import pytest
from aiohttp.test_utils import TestClient, TestServer

from janus_graph.config import JanusSettings
from janus_graph.daemon.episode_queue_adapter import (
    EpisodeDuplicateError,
    EpisodeQueueAdapter,
)
from janus_graph.daemon.falkordb_supervisor import DaemonSupervisor
from janus_graph.daemon.http_server import __version__, build_app
from janus_graph.daemon.lifespan import DaemonContext
from janus_graph.pipeline.queue import EpisodeQueue


# ─── fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def tmp_queue_path(tmp_path):
    """Per-test SQLite queue path so tests don't share state."""
    return str(tmp_path / "episodes.db")


@pytest.fixture
def settings(tmp_queue_path) -> JanusSettings:
    """Daemon settings with the lock opened up (the gate is the subject here)."""
    s = JanusSettings()
    s.pipeline.queue_db_path = tmp_queue_path
    s.daemon.lock.mode = "daemon_only"
    return s


async def _make_adapter(settings: JanusSettings, dedup_mode) -> EpisodeQueueAdapter:
    """Build an adapter wired to a fresh queue, with an explicit mode."""
    queue = EpisodeQueue(db_path=settings.pipeline.queue_db_path)
    adapter = EpisodeQueueAdapter(
        queue=queue,
        lock_settings=settings.daemon.lock,
        dedup_mode=dedup_mode,
    )
    await adapter.start()
    return adapter


def _rows(adapter: EpisodeQueueAdapter):
    """Return every episode row as dicts (newest last)."""
    with sqlite3.connect(str(adapter._queue.db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return [
            dict(r)
            for r in conn.execute(
                "SELECT id, status, payload_hash, canonical_hash, dedup_gate "
                "FROM episodes ORDER BY created_at, id"
            )
        ]


def _ledger(adapter: EpisodeQueueAdapter):
    """Return the durable dedup_events ledger rows."""
    with sqlite3.connect(str(adapter._queue.db_path)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT mode, action, existing_episode_id, writer FROM dedup_events ORDER BY id"
                )
            ]
        except sqlite3.Error:
            return []


# ─── the mode resolution ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mode_off_does_not_reject_duplicate(settings):
    """off: the duplicate is ACCEPTED. The gate is not applied at all."""
    adapter = await _make_adapter(settings, "off")
    r1 = await adapter.enqueue_guarded(content="dup-off", group_id="graphiti_memory")
    r2 = await adapter.enqueue_guarded(content="dup-off", group_id="graphiti_memory")

    assert r2.episode_id != r1.episode_id, "off must accept a duplicate"
    rows = _rows(adapter)
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_mode_off_stamps_gate_off_and_no_legacy_hash(settings):
    """off: ``dedup_gate='off'`` and NO legacy payload_hash.

    The legacy column is stamped only when the mode is not off — same
    contract as the MCP writer. The CANONICAL hash is a content ADDRESS
    (not a gate), so it is always stamped: the processing-layer backstop
    asks "was this exact content already ingested?", and a row with no
    hash at all would be invisible to it.
    """
    adapter = await _make_adapter(settings, "off")
    await adapter.enqueue_guarded(content="off-stamp", group_id="graphiti_memory")

    row = _rows(adapter)[0]
    assert row["dedup_gate"] == "off"
    assert row["payload_hash"] is None, "off must not stamp the legacy hash"
    assert row["canonical_hash"], "the canonical content address is always stamped"


@pytest.mark.asyncio
async def test_mode_off_writes_no_ledger_row(settings):
    """off: nothing is DETECTED, so nothing is recorded.

    A ledger row in off mode would be a false statement: it would say a
    duplicate was seen by a gate that was not running.
    """
    adapter = await _make_adapter(settings, "off")
    await adapter.enqueue_guarded(content="dup-off-ledger", group_id="graphiti_memory")
    await adapter.enqueue_guarded(content="dup-off-ledger", group_id="graphiti_memory")

    assert _ledger(adapter) == []


@pytest.mark.asyncio
async def test_mode_observe_accepts_duplicate_but_records_it(settings):
    """observe: DETECTED and RECORDED, but ACCEPTED.

    Rejecting is the one thing observe must never do — that is the
    whole point of running observe first: learn what the gate WOULD have
    caught without dropping anything.
    """
    adapter = await _make_adapter(settings, "observe")
    r1 = await adapter.enqueue_guarded(content="dup-observe", group_id="graphiti_memory")
    r2 = await adapter.enqueue_guarded(content="dup-observe", group_id="graphiti_memory")

    assert r2.episode_id != r1.episode_id, "observe must NOT reject"
    assert len(_rows(adapter)) == 2, "both rows must be enqueued"

    events = _ledger(adapter)
    assert len(events) == 1, f"expected exactly one ledger row, got {events}"
    assert events[0]["action"] == "detected"
    assert events[0]["mode"] == "observe"
    assert events[0]["existing_episode_id"] == r1.episode_id


@pytest.mark.asyncio
async def test_mode_observe_stamps_gate_observe(settings):
    """observe: the row is stamped ``dedup_gate='observe'``.

    The UNIQUE partial index only covers ``dedup_gate='enforce'``. So an
    observe row is deliberately OUTSIDE the index's protection — the DB
    must not reject a duplicate that the policy says must be accepted.
    """
    adapter = await _make_adapter(settings, "observe")
    await adapter.enqueue_guarded(content="observe-stamp", group_id="graphiti_memory")

    row = _rows(adapter)[0]
    assert row["dedup_gate"] == "observe"
    assert row["payload_hash"], "observe stores the hash (it must be able to detect)"


@pytest.mark.asyncio
async def test_mode_observe_duplicate_is_not_blocked_by_unique_index(settings):
    """Two observe rows with the same hash must BOTH survive.

    This is the DB-level half of observe's contract. The partial UNIQUE
    index predicate is ``dedup_gate='enforce'``, so an observe-stamped
    row is not covered and the INSERT cannot fail.
    """
    adapter = await _make_adapter(settings, "observe")
    await adapter.enqueue_guarded(content="same-hash", group_id="graphiti_memory")
    await adapter.enqueue_guarded(content="same-hash", group_id="graphiti_memory")

    rows = _rows(adapter)
    assert len(rows) == 2
    assert rows[0]["canonical_hash"] == rows[1]["canonical_hash"]
    assert {r["dedup_gate"] for r in rows} == {"observe"}


@pytest.mark.asyncio
async def test_mode_enforce_rejects_and_records(settings):
    """enforce: DETECTED, RECORDED and REJECTED."""
    adapter = await _make_adapter(settings, "enforce")
    r1 = await adapter.enqueue_guarded(content="dup-enforce", group_id="graphiti_memory")

    with pytest.raises(EpisodeDuplicateError) as ei:
        await adapter.enqueue_guarded(content="dup-enforce", group_id="graphiti_memory")

    assert ei.value.existing_episode_id == r1.episode_id
    assert len(_rows(adapter)) == 1, "the duplicate must NOT be enqueued"

    events = _ledger(adapter)
    assert len(events) == 1
    assert events[0]["action"] == "rejected"
    assert events[0]["mode"] == "enforce"


@pytest.mark.asyncio
async def test_mode_enforce_stamps_gate_enforce(settings):
    """enforce: the row is stamped ``dedup_gate='enforce'``.

    That stamp is what puts the row INSIDE the UNIQUE partial index, which
    is what closes the cross-process race between the daemon and the MCP
    writer.
    """
    adapter = await _make_adapter(settings, "enforce")
    await adapter.enqueue_guarded(content="enforce-stamp", group_id="graphiti_memory")

    row = _rows(adapter)[0]
    assert row["dedup_gate"] == "enforce"
    assert row["payload_hash"]


@pytest.mark.asyncio
async def test_unknown_mode_folds_to_off_and_never_rejects(settings):
    """A typo must never start DROPPING episodes.

    Falling back to ``enforce`` on an unrecognised value would turn a
    config typo into silent data loss. Falling back to ``off`` only means
    the gate is not applied — a safe, visible no-op.
    """
    adapter = await _make_adapter(settings, "enforse")
    await adapter.enqueue_guarded(content="typo", group_id="graphiti_memory")
    r2 = await adapter.enqueue_guarded(content="typo", group_id="graphiti_memory")

    assert r2.episode_id, "an unknown mode must not reject"
    assert _rows(adapter)[0]["dedup_gate"] == "off"


@pytest.mark.asyncio
async def test_per_call_override_wins_over_the_adapter_mode(settings):
    """A per-call mode overrides the adapter's configured mode."""
    adapter = await _make_adapter(settings, "enforce")
    # The adapter is in enforce, but this call says observe -> no reject.
    r1 = await adapter.enqueue_guarded(
        content="override", group_id="graphiti_memory", dedup_mode="observe"
    )
    r2 = await adapter.enqueue_guarded(
        content="override", group_id="graphiti_memory", dedup_mode="observe"
    )
    assert r2.episode_id != r1.episode_id

    # And the reverse: the adapter is in observe, the call says enforce.
    adapter2 = await _make_adapter(settings, "observe")
    await adapter2.enqueue_guarded(content="override2", group_id="graphiti_memory")
    with pytest.raises(EpisodeDuplicateError):
        await adapter2.enqueue_guarded(
            content="override2", group_id="graphiti_memory", dedup_mode="enforce"
        )


@pytest.mark.asyncio
async def test_adapter_falls_back_to_the_config_file_when_no_mode_given(settings, monkeypatch):
    """With no explicit mode, the adapter reads ``cfg.pipeline.dedup.mode``.

    This is the wiring the daemon uses at boot — the adapter is handed
    the value from the daemon's own settings, which come from the same
    config file the MCP stdio writer reads.
    """
    from janus_graph.pipeline import queue as q_mod

    def _fake_load_config():
        cfg = JanusSettings()
        cfg.pipeline.dedup.mode = "enforce"
        return cfg

    q_mod.reset_dedup_mode_cache()
    try:
        monkeypatch.setattr("janus_graph.config.load_config", _fake_load_config)
        queue = EpisodeQueue(db_path=settings.pipeline.queue_db_path)
        adapter = EpisodeQueueAdapter(queue=queue, lock_settings=settings.daemon.lock)
        await adapter.start()

        await adapter.enqueue_guarded(content="from-config", group_id="graphiti_memory")
        with pytest.raises(EpisodeDuplicateError):
            await adapter.enqueue_guarded(content="from-config", group_id="graphiti_memory")
    finally:
        q_mod.reset_dedup_mode_cache()


# ─── the HTTP layer ─────────────────────────────────────────────────────


async def _http_client(settings: JanusSettings, dedup_mode) -> TestClient:
    queue = EpisodeQueue(db_path=settings.pipeline.queue_db_path)
    adapter = EpisodeQueueAdapter(
        queue=queue,
        lock_settings=settings.daemon.lock,
        dedup_mode=dedup_mode,
    )
    await adapter.start()
    ctx = DaemonContext(
        settings=settings,
        supervisor=DaemonSupervisor(
            engine_config=settings.engine,
            circuit_config=settings.daemon.falkor_circuit,
        ),
        http_settings=settings.daemon.http,
        version=__version__,
        shutdown_event=asyncio.Event(),
        queue=queue,
        queue_adapter=adapter,
    )
    app = build_app(ctx)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_http_returns_409_only_in_enforce_mode(settings):
    """The HTTP status must follow the mode: 409 in enforce, 202 otherwise."""
    for mode, expected_status in (
        ("off", 202),
        ("observe", 202),
        ("enforce", 409),
    ):
        client = await _http_client(settings, mode)
        try:
            body = {"content": f"http-dup-{mode}", "name": f"t-{mode}"}
            r1 = await client.post("/episodes", json=body)
            r2 = await client.post("/episodes", json=body)
            assert r1.status == 202, f"{mode}: first POST must be accepted"
            assert r2.status == expected_status, (
                f"{mode}: second POST must be {expected_status}, got {r2.status}"
            )
            if expected_status == 409:
                payload = await r2.json()
                assert payload["code"] == "DUPLICATE_EPISODE"
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_http_202_body_reports_the_mode(settings):
    """The 202 body must not claim a dedup hit that did not happen."""
    client = await _http_client(settings, "observe")
    try:
        body = {"content": "http-observe-body", "name": "t-body"}
        await client.post("/episodes", json=body)
        resp = await client.post("/episodes", json=body)
        assert resp.status == 202
        payload = await resp.json()
        assert payload["deduplicated"] is False
        assert payload["episode_id"]
    finally:
        await client.close()


# ─── the YAML 1.1 boolean trap ─────────────────────────────────────────
#
# PyYAML implements YAML 1.1: an unquoted `off` is the BOOLEAN False, not
# the string "off". Pydantic v2 refuses to coerce a bool into a str
# field, so `pipeline.dedup.mode: off` used to make JanusSettings()
# RAISE — and every process that builds the settings at boot (MCP stdio
# server, daemon, hook python) died with it. The SAFE mode was the one
# you could not set.
#
# The shipped config.example.yaml already quotes "off", so the trap only
# fires when an operator writes the mode unquoted. These tests pin the
# fix: the validator runs mode="before" and folds the booleans back.


@pytest.mark.parametrize(
    "yaml_value,expected",
    [
        ("off", "off"),
        ("on", "observe"),
        ("yes", "observe"),
        ("no", "off"),
        ("enforce", "enforce"),
        ("observe", "observe"),
        ('"off"', "off"),
    ],
)
def test_yaml_mode_values_resolve_without_raising(yaml_value, expected):
    """Every spelling of the mode must load — none may crash the boot."""
    import yaml

    doc = yaml.safe_load(
        "pipeline:" + chr(10) + "  dedup:" + chr(10) + "    mode: " + yaml_value + chr(10)
    )
    s = JanusSettings(**doc)
    assert s.pipeline.dedup.mode == expected


def test_unquoted_off_does_not_crash_the_boot(tmp_path):
    """The regression this fixes: `mode: off` raised a ValidationError."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("pipeline:" + chr(10) + "  dedup:" + chr(10) + "    mode: off" + chr(10))

    os.environ["JANUS_CONFIG_PATH"] = str(cfg)
    try:
        from janus_graph.config import load_config

        resolved = load_config()
        assert resolved.pipeline.dedup.mode == "off"
    finally:
        os.environ.pop("JANUS_CONFIG_PATH", None)


def test_a_yaml_boolean_can_never_enable_enforce():
    """An unquoted YAML boolean must not be able to turn a REJECTING gate on.

    `off`/`on` are ambiguous in YAML 1.1. The fold picks the safe
    reading for both: False -> "off" (no gate), True -> "observe" (detect,
    never reject). Only the literal string "enforce" can reject.
    """
    import yaml

    for boolean_value in (True, False):
        doc = yaml.safe_load(
            "pipeline:"
            + chr(10)
            + "  dedup:"
            + chr(10)
            + "    mode: "
            + str(boolean_value)
            + chr(10)
        )
        s = JanusSettings(**doc)
        assert s.pipeline.dedup.mode != "enforce"
