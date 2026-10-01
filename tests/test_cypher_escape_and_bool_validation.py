"""Regression tests for two bugs found 2026-10-01 in daemon/search_engine.py.

BUG #1 — Cypher string escaping used SQL-style quote-doubling (``''``).
  FalkorDB v8.9.x uses BACKSLASH escaping inside ``'...'``. Verified live
  against graphiti_memory on :6379:
      quote-doubling  -> 5 exact, 2 SILENTLY CORRUPTED, 5 parse errors
      backslash-first -> 12/12 exact, 0 corrupted, 0 errors
  Two distinct failure modes, not one:
    (a) parse error  — ``'O'Brien'`` lexes as ``'O'`` + bare identifier
    (b) corruption   — ``'a\\b'`` is read as escape ``\\b`` -> ``'a\x08'``
  The parse error was then rewrapped as FALKOR_DOWN (503), telling
  monitoring "database down" for what is really a malformed query.

BUG #2 — validate_request accepted JSON ``true`` as a number.
  ``bool`` subclasses ``int`` in Python, so ``max_hops: true`` passed the
  int check and became hops=1, ``min_cosine: true`` became 1.0 and
  rejected every real result, ``mmr_lambda: true`` became 1.0 and disabled
  MMR diversity.

  Reachability caveat (verified 2026-10-01 against the live daemon):
  ``/search/graph`` is registered with ``add_get`` ONLY, and its GET
  parser hands values over as raw strings — so a query string can never
  deliver a Python ``bool`` into the payload. The unreachable POST body
  branch was removed, which means **no HTTP route can reach these guards
  today**. They are kept because ``SearchEngine`` is a library and direct
  (non-HTTP) callers can still pass a real bool. Do not expect a live
  HTTP probe to exercise this section.

Pure unit tests — no live FalkorDB, no network. The live round-trip
evidence that motivated this fix is recorded in the docstring of
``_escape_cypher_string``.
"""
import pytest

from janus_graph.daemon.search_engine import (
    _cypher_param,
    _escape_cypher_string,
    _classify_cypher_error,
    SearchValidationError,
)

Q = chr(39)
BS = chr(92)

HOSTILE_SEEDS = [
    "O'Brien",
    "a" + BS + "b",
    "both" + Q + BS + "mix",
    "trail" + BS,
    "quote" + Q + Q + "double",
    "plain",
]


# ─── BUG #1a: escaping produces a parseable literal ───────────────────


class TestEscaping:
    @pytest.mark.parametrize("raw", HOSTILE_SEEDS)
    def test_apostrophe_is_backslash_escaped_not_doubled(self, raw):
        out = _escape_cypher_string(raw)
        # Never SQL-style doubling — that is what FalkorDB rejects.
        assert Q + Q not in out, "quote-doubling is invalid in this engine"

    def test_obrien_becomes_backslash_quote(self):
        assert _escape_cypher_string("O'Brien") == "O" + BS + Q + "Brien"

    def test_backslash_is_escaped_before_quote(self):
        # Order matters: escaping quotes first would turn the backslash
        # introduced by quote-escaping into an escaped backslash.
        assert _escape_cypher_string(BS) == BS + BS
        assert _escape_cypher_string(BS + Q) == BS + BS + BS + Q

    def test_trailing_backslash_does_not_swallow_closing_quote(self):
        # 'trail\' would escape the closing quote and run off the literal.
        assert _escape_cypher_string("trail" + BS) == "trail" + BS + BS

    def test_corrupting_backslash_b_is_preserved(self):
        # Regression: naive escaping turned 'a\\b' into 'a\x08'.
        assert _escape_cypher_string("a" + BS + "b") == "a" + BS + BS + "b"

    def test_plain_string_unchanged(self):
        assert _escape_cypher_string("Maple Aikon") == "Maple Aikon"

    def test_cypher_param_uses_backslash_escaping(self):
        assert _cypher_param("O'Brien") == Q + "O" + BS + Q + "Brien" + Q

    def test_cypher_param_backslash(self):
        assert _cypher_param("a" + BS + "b") == Q + "a" + BS + BS + "b" + Q

    def test_cypher_param_non_strings_unaffected(self):
        assert _cypher_param(None) == "NULL"
        assert _cypher_param(True) == "true"
        assert _cypher_param(False) == "false"
        assert _cypher_param(3) == "3"
        assert _cypher_param(0.5) == "0.5"


# ─── BUG #1b: parse errors must not masquerade as an outage ───────────


class TestErrorClassification:
    def test_parse_error_becomes_invalid_cypher_not_503(self):
        err = _classify_cypher_error(
            Exception("errMsg: Invalid input " + Q + ": expected " + Q)
        )
        assert err.code == "INVALID_CYPHER"
        assert err.code != "FALKOR_DOWN"

    @pytest.mark.parametrize(
        "msg",
        [
            "Connection refused",
            "Error 111 connecting to 127.0.0.1:6379",
            "timed out",
            "LOADING graph is not ready",
        ],
    )
    def test_real_outages_still_report_falkor_down(self, msg):
        # Must not mask a genuine outage as a client error.
        assert _classify_cypher_error(Exception(msg)).code == "FALKOR_DOWN"

    def test_connection_error_object(self):
        assert _classify_cypher_error(ConnectionError("refused")).code == "FALKOR_DOWN"


# ─── BUG #2: bool must not pass as a number ───────────────────────────


def _engine():
    """Build a real SearchEngine. validate_request never touches redis, so a
    mocked embed client is enough. Mirrors the helper in test_search_graph.py.
    """
    from unittest.mock import MagicMock

    from janus_graph.config import JanusSettings
    from janus_graph.daemon.search_engine import SearchEngine

    s = JanusSettings()
    return SearchEngine(
        engine_config=s.engine,
        search_settings=s.daemon.search_graph,
        embedding_client=MagicMock(),
    )


def _payload(**kw):
    base = {"seed_entities": ["Maple Aikon"]}
    base.update(kw)
    return base


class TestBoolRejected:
    def test_max_hops_true_rejected(self):
        with pytest.raises(SearchValidationError) as ei:
            _engine().validate_request(_payload(max_hops=True))
        assert ei.value.code == "INVALID_HOPS"

    def test_limit_true_rejected(self):
        with pytest.raises(SearchValidationError) as ei:
            _engine().validate_request(_payload(limit=True))
        assert ei.value.code == "INVALID_LIMIT"

    def test_min_cosine_true_rejected(self):
        # True -> 1.0 would reject every real cosine result.
        with pytest.raises(SearchValidationError) as ei:
            _engine().validate_request(_payload(min_cosine=True))
        assert ei.value.code == "MIN_COSINE_OUT_OF_RANGE"

    def test_mmr_lambda_true_falls_back_to_default(self):
        # Out-of-range mmr_lambda degrades to default by design; a bool must
        # not silently become 1.0 and switch MMR diversity off.
        eng = _engine()
        out = eng.validate_request(_payload(mmr_lambda=True))
        assert out["mmr_lambda"] == eng._settings.default_mmr_lambda
        assert out["mmr_lambda"] != 1.0

    def test_valid_numbers_still_accepted(self):
        out = _engine().validate_request(
            _payload(max_hops=2, limit=5, min_cosine=0.3, mmr_lambda=0.7)
        )
        assert out["max_hops"] == 2
        assert out["limit"] == 5
        assert out["min_cosine"] == 0.3
        assert out["mmr_lambda"] == 0.7

    def test_int_one_still_accepted_for_hops_and_limit(self):
        # Guard against over-correction: 1 is a legitimate int.
        out = _engine().validate_request(_payload(max_hops=1, limit=1))
        assert out["max_hops"] == 1
        assert out["limit"] == 1

    def test_bool_still_accepted_for_include_episodes(self):
        # include_episodes is genuinely boolean — must keep working.
        out = _engine().validate_request(_payload(include_episodes=True))
        assert out["include_episodes"] is True
