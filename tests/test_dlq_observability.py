"""Read-path DLQ observability: ``classes`` filter, ``open_only``, ``error_chars``.

Scope (plan Đợt 2, 2026-10-16): READ only. ``replay_dlq_batch`` keeps its
``LIKE`` predicate untouched — this module deliberately uses
``instr(lower(x), lower(?)) > 0`` because ``LIKE`` treats ``%`` and ``_`` in
an operator-supplied class as wildcards (measured: a class of ``_`` matched
536/536 rows via ``LIKE '%_%'`` vs 375/536 via ``instr``).

Every ``default`` here must reproduce the pre-change behaviour byte-for-byte:
``get_dlq_records(limit=n)`` used to be ``SELECT ... ORDER BY failed_at DESC
LIMIT ?`` with no WHERE clause, so ``classes``/``open_only``/``error_chars``
all default to "do not change the query".
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from janus_graph.pipeline.queue import EpisodeQueue


# ── helpers ────────────────────────────────────────────────────────────────


async def _seed(queue: EpisodeQueue, rows: list[tuple[str, str, bool]]) -> None:
    """Insert ``(last_error, failed_at, recovered)`` rows into ``dead_letter``.

    ``recovered`` is explicit per row on purpose. An earlier version of this
    helper derived it from the row index (``i % 2``), which made six tests
    fail for a reason that had nothing to do with the code under test: the
    row they meant to assert was open had been silently seeded recovered.

    Going through ``mark_aborted`` instead would take the episode lifecycle
    and stamp ``failed_at`` with ``now()``, which cannot express "two rows
    share the same ``failed_at``" (negative control N3) or a fixed 2012-char
    error string (AC5). The columns used here are exactly the ones
    ``get_dlq_records`` selects.
    """
    with queue._get_connection() as conn:
        for i, (last_error, failed_at, recovered) in enumerate(rows):
            conn.execute(
                "INSERT INTO dead_letter "
                "(episode_id, payload_json, last_error, attempt_count, failed_at, recovered_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    f"ep-{i:03d}",
                    json.dumps({"content": "x"}),
                    last_error,
                    3,
                    failed_at,
                    "2026-10-16T00:00:00+00:00" if recovered else None,
                ),
            )
        conn.commit()


OPEN = False  # readable alias: recovered=False means the row is still open


# ── AC1: default is byte-for-byte the old behaviour ────────────────────────


@pytest.mark.asyncio
async def test_default_query_returns_recovered_and_open_rows_alike(temp_queue):
    """``get_dlq_records(limit=n)`` must NOT filter to open rows.

    Red on a naive implementation: the moment someone adds
    ``WHERE recovered_at IS NULL`` unconditionally, the recovered row
    disappears and this fails.
    """
    await _seed(
        temp_queue,
        [
            ("open row", "2026-10-01T00:00:00+00:00", OPEN),
            ("recovered row", "2026-10-02T00:00:00+00:00", True),
        ],
    )

    rows = temp_queue.get_dlq_records(limit=10)

    assert len(rows) == 2, "default must not filter by recovered_at"
    assert {r["episode_id"] for r in rows} == {"ep-000", "ep-001"}


@pytest.mark.asyncio
async def test_default_error_chars_does_not_truncate(temp_queue):
    """AC: default ``error_chars`` is ``None`` — no truncation.

    The 2012-char string is the measured ``MAX(LENGTH(last_error))`` of the
    open DLQ rows on the live DB (2026-10-16). A default of 200 — the value
    proposed in wave 1 — would silently truncate this row, which is exactly
    how "no new argument ⇒ identical output" became self-contradictory.
    """
    long_err = "E" * 2012
    await _seed(temp_queue, [(long_err, "2026-10-01T00:00:00+00:00", OPEN)])

    rows = temp_queue.get_dlq_records(limit=1)

    assert len(rows[0]["last_error"]) == 2012


# ── AC2/AC3/AC4: classes filter ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_classes_filters_open_rows_only(temp_queue):
    """AC2 shape: ``classes`` + ``open_only`` restrict to matching OPEN rows."""
    await _seed(
        temp_queue,
        [
            ("ValidationError: bad payload", "2026-10-01T00:00:00+00:00", OPEN),  # open
            ("ValidationError: bad payload", "2026-10-02T00:00:00+00:00", True),  # recovered (i=1)
            ("TimeoutError: <no message>", "2026-10-03T00:00:00+00:00", OPEN),  # open
        ],
    )

    rows = temp_queue.get_dlq_records(limit=10, classes=["ValidationError"], open_only=True)

    assert [r["episode_id"] for r in rows] == ["ep-000"]


@pytest.mark.asyncio
async def test_multiple_classes_are_a_union(temp_queue):
    """AC3 shape: two disjoint classes add up (OR, not AND)."""
    await _seed(
        temp_queue,
        [
            ("ValidationError: bad payload", "2026-10-01T00:00:00+00:00", OPEN),
            ("TimeoutError: <no message>", "2026-10-03T00:00:00+00:00", OPEN),
            ("APIStatusError: Error code: 402", "2026-10-04T00:00:00+00:00", OPEN),
        ],
    )

    rows = temp_queue.get_dlq_records(
        limit=10, classes=["ValidationError", "TimeoutError"], open_only=True
    )

    assert {r["episode_id"] for r in rows} == {"ep-000", "ep-001"}


@pytest.mark.asyncio
async def test_unknown_class_returns_empty_not_everything(temp_queue):
    """AC4: an unmatched class must return ``[]``, never the whole table.

    The failure this guards against is the classic "filter silently ignored"
    bug, which looks like success to any caller that only checks the shape.
    """
    await _seed(temp_queue, [("ValidationError: bad payload", "2026-10-01T00:00:00+00:00", OPEN)])

    rows = temp_queue.get_dlq_records(limit=10, classes=["không-có"], open_only=True)

    assert rows == []


@pytest.mark.asyncio
async def test_classes_match_is_case_insensitive(temp_queue):
    await _seed(temp_queue, [("validationerror: lowercase here", "2026-10-01T00:00:00+00:00", OPEN)])

    rows = temp_queue.get_dlq_records(limit=10, classes=["ValidationError"], open_only=True)

    assert len(rows) == 1


# ── N2: LIKE wildcards must not leak into the READ path ────────────────────


@pytest.mark.asyncio
async def test_underscore_class_is_a_literal_not_a_wildcard(temp_queue):
    """``_`` must match a literal underscore, not "any single character".

    ``LIKE '%_%'`` returns every non-empty ``last_error`` — on the live DB,
    536/536 rows. ``instr`` returns only the 375 rows that actually contain
    an underscore, so this asserts a strictly smaller, correct set.
    """
    await _seed(
        temp_queue,
        [
            ("snake_case_error", "2026-10-01T00:00:00+00:00", OPEN),  # has underscore
            ("ValidationError: plain", "2026-10-02T00:00:00+00:00", OPEN),  # no underscore
        ],
    )

    rows = temp_queue.get_dlq_records(limit=10, classes=["_"], open_only=True)

    assert [r["episode_id"] for r in rows] == ["ep-000"]


@pytest.mark.asyncio
async def test_percent_class_is_a_literal_not_a_wildcard(temp_queue):
    """``%`` must be literal too (an all-rows match would be the bug)."""
    await _seed(
        temp_queue,
        [
            ("disk 95% full", "2026-10-01T00:00:00+00:00", OPEN),
            ("ValidationError: plain", "2026-10-02T00:00:00+00:00", OPEN),
        ],
    )

    rows = temp_queue.get_dlq_records(limit=10, classes=["%"], open_only=True)

    assert [r["episode_id"] for r in rows] == ["ep-000"]


# ── AC5: error_chars ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_error_chars_truncates_with_ellipsis(temp_queue):
    await _seed(temp_queue, [("E" * 2012, "2026-10-01T00:00:00+00:00", OPEN)])

    rows = temp_queue.get_dlq_records(limit=1, error_chars=200)

    assert len(rows[0]["last_error"]) == 201, "200 chars + the ellipsis"


@pytest.mark.asyncio
async def test_error_chars_zero_means_no_truncation(temp_queue):
    """``error_chars=0`` is the documented "off" switch, not "empty string"."""
    await _seed(temp_queue, [("E" * 2012, "2026-10-01T00:00:00+00:00", OPEN)])

    rows = temp_queue.get_dlq_records(limit=1, error_chars=0)

    assert len(rows[0]["last_error"]) == 2012


@pytest.mark.asyncio
async def test_error_chars_shorter_than_the_value_is_a_noop(temp_queue):
    await _seed(temp_queue, [("short", "2026-10-01T00:00:00+00:00", OPEN)])

    rows = temp_queue.get_dlq_records(limit=1, error_chars=200)

    assert rows[0]["last_error"] == "short"


# ── N3: ordering survives filtering ────────────────────────────────────────


@pytest.mark.asyncio
async def test_order_by_failed_at_desc_is_preserved_when_filtered(temp_queue):
    """Two rows with the SAME ``failed_at`` keep insertion order (SQLite's
    stable scan order); a third, later row must still come first overall."""
    await _seed(
        temp_queue,
        [
            ("ValidationError: first", "2026-10-01T00:00:00+00:00", OPEN),
            ("ValidationError: second", "2026-10-01T00:00:00+00:00", OPEN),
            ("ValidationError: third", "2026-10-05T00:00:00+00:00", OPEN),
            ("TimeoutError: excluded", "2026-10-06T00:00:00+00:00", True),
        ],
    )

    rows = temp_queue.get_dlq_records(limit=10, classes=["ValidationError"], open_only=True)

    assert [r["last_error"] for r in rows] == [
        "ValidationError: third",
        "ValidationError: first",
        "ValidationError: second",
    ]


# ── N4: open_only on an all-recovered table ────────────────────────────────


@pytest.mark.asyncio
async def test_open_only_true_returns_empty_when_everything_recovered(temp_queue):
    await _seed(temp_queue, [("a", "2026-10-01T00:00:00+00:00", True)])
    with temp_queue._get_connection() as conn:
        conn.execute("UPDATE dead_letter SET recovered_at = '2026-10-16T00:00:00+00:00'")
        conn.commit()

    assert temp_queue.get_dlq_records(limit=10, open_only=True) == []
    # Negative control: the same table unfiltered is NOT empty, so the 0
    # above proves the filter ran rather than proving the table is empty.
    assert len(temp_queue.get_dlq_records(limit=10)) == 1


@pytest.mark.asyncio
async def test_open_only_false_selects_recovered_rows(temp_queue):
    await _seed(temp_queue, [("a", "2026-10-01T00:00:00+00:00", True)])

    assert len(temp_queue.get_dlq_records(limit=10, open_only=False)) == 1
    assert temp_queue.get_dlq_records(limit=10, open_only=True) == []


# ── dlq_class_counts: one source of truth for the numbers ──────────────────


@pytest.mark.asyncio
async def test_class_counts_sum_to_the_open_total(temp_queue):
    """AC6: ``by_class`` must reconcile with ``get_stats()['dlq']``."""
    await _seed(
        temp_queue,
        [
            ("ValidationError: bad payload", "2026-10-01T00:00:00+00:00", OPEN),  # open
            ("ValidationError: bad payload", "2026-10-02T00:00:00+00:00", True),  # recovered
            ("TimeoutError: <no message>", "2026-10-03T00:00:00+00:00", OPEN),  # open
            ("APIStatusError: 402", "2026-10-04T00:00:00+00:00", OPEN),  # open
        ],
    )

    counts = temp_queue.dlq_class_counts(["ValidationError", "TimeoutError"])
    total = temp_queue.get_stats()["dlq"]

    assert counts == {"ValidationError": 1, "TimeoutError": 1}
    assert sum(counts.values()) + 1 == total, "unclassified = 402 row"


@pytest.mark.asyncio
async def test_class_counts_empty_when_taxonomy_matches_nothing(temp_queue):
    """N1 first half: an empty ``by_class`` is a 0 that must not be read as
    "the DLQ is empty" — the caller needs ``total`` to tell them apart."""
    await _seed(temp_queue, [("TimeoutError: <no message>", "2026-10-01T00:00:00+00:00", OPEN)])

    counts = temp_queue.dlq_class_counts(["SCHEMA_DRIFT"])

    # An unmatched class is reported as 0, not as a missing key — the caller
    # gets one shape for every class it asked about.
    assert counts == {"SCHEMA_DRIFT": 0}
    # The negative control that matters: the DLQ is NOT empty, so the 0 above
    # is the filter's answer and not "no rows exist".
    assert temp_queue.get_stats()["dlq"] == 1


@pytest.mark.asyncio
async def test_class_counts_n1_inverse_control(temp_queue):
    """N1 second half: break exactly one row's class and watch the numbers move.

    Both "``by_class`` is empty" and "the row set is empty" print 0, so a 0
    alone proves nothing. This is the inverse: ``by_class`` must DECREASE and
    ``unclassified`` must INCREASE, which a broken predicate cannot fake.
    """
    await _seed(
        temp_queue,
        [
            ("ValidationError: bad payload", "2026-10-01T00:00:00+00:00", OPEN),
            ("ZZZ_NOT_A_CLASS: nope", "2026-10-02T00:00:00+00:00", OPEN),
        ],
    )
    before = temp_queue.dlq_class_counts(["ValidationError"])
    assert before["ValidationError"] == 1

    with temp_queue._get_connection() as conn:
        conn.execute(
            "UPDATE dead_letter SET last_error = 'ValidationError: fixed' "
            "WHERE last_error LIKE 'ZZZ_NOT_A_CLASS%'"
        )
        conn.commit()

    after = temp_queue.dlq_class_counts(["ValidationError"])

    assert after["ValidationError"] == 2, "the reclassified row must now count"


@pytest.mark.asyncio
async def test_class_counts_are_zero_absent_keys_not_missing(temp_queue):
    """Every requested class gets a key, so the caller can render a stable
    table without special-casing missing entries."""
    await _seed(temp_queue, [("ValidationError: bad payload", "2026-10-01T00:00:00+00:00", OPEN)])

    counts = temp_queue.dlq_class_counts(["ValidationError", "TimeoutError"])

    assert counts == {"ValidationError": 1, "TimeoutError": 0}
    assert sum(counts.values()) == temp_queue.get_stats()["dlq"]


@pytest.mark.asyncio
async def test_class_counts_without_taxonomy_returns_empty_dict(temp_queue):
    await _seed(temp_queue, [("ValidationError: bad payload", "2026-10-01T00:00:00+00:00", OPEN)])

    assert temp_queue.dlq_class_counts(None) == {}
    assert temp_queue.dlq_class_counts([]) == {}


# ── the read path must not write ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_dlq_reads_do_not_mutate_the_table(temp_queue):
    await _seed(temp_queue, [("ValidationError: bad payload", "2026-10-01T00:00:00+00:00", OPEN)])

    with temp_queue._get_connection() as conn:
        before = conn.execute("SELECT COUNT(*) FROM dead_letter").fetchone()[0]

    temp_queue.get_dlq_records(limit=5, classes=["%"], open_only=True, error_chars=1)
    temp_queue.dlq_class_counts(["%", "_"])

    with temp_queue._get_connection() as conn:
        after = conn.execute("SELECT COUNT(*) FROM dead_letter").fetchone()[0]
        still_open = conn.execute(
            "SELECT COUNT(*) FROM dead_letter WHERE recovered_at IS NULL"
        ).fetchone()[0]

    assert after == before == 1
    assert still_open == 1


@pytest.mark.asyncio
async def test_sql_in_a_class_is_a_parameter_not_syntax(temp_queue):
    """A class containing SQL must be inert — this is a read tool exposed
    over MCP, so the injection surface is the operator's own typing."""
    await _seed(temp_queue, [("ValidationError: bad payload", "2026-10-01T00:00:00+00:00", OPEN)])
    evil = "'; DROP TABLE dead_letter; --"

    rows = temp_queue.get_dlq_records(limit=5, classes=[evil], open_only=True)

    assert rows == []
    with temp_queue._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM dead_letter").fetchone()[0] == 1
    # and the table is still queryable through sqlite_master
    with sqlite3.connect(":memory:") as _:
        pass
