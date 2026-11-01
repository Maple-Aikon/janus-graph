#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 integration smoke gates (Plan #2 §6 verification).
#
# Gates:
#   T1 — happy path: Falkor alive → GET /health returns 200 + status=ready
#   T6 — Falkor down: GET /health returns 503 + status=degraded
#   T8 — circuit-breaker: 3 failures → circuit_open, 0 probe log in cooldown
#   T9 — OPEN doesn't break T6: while circuit OPEN, /health still 503 with body
#
# Usage:
#   bash tests/smoke_phase2.sh           # full gate sweep
#   bash tests/smoke_phase2.sh t1        # single gate
#
# Requires:
#   - python3 with janus_graph installed + aiohttp
#   - FalkorDB binaries present (./bin/redis-server, ./bin/falkordb.so)
#   - daemon NOT already running on port 8765 (script kills any leftover)
#
# Side effects: starts/stops daemon process, starts/stops falkordb.
# Run from repo root. Idempotent cleanup on exit (best-effort).
# ─────────────────────────────────────────────────────────────────────────────
#
# ─── EXTERNAL-DB MODE (JANUS_EXTERNAL_FALKOR=1) ───────────────────────────────
# Added 2026-11-01 so this gate can run in CI, where the database is a
# GitHub Actions *service container* owned by the runner.
#
# In this mode the script does NOT start, stop or kill any FalkorDB process.
# It talks to a database somebody else owns, at the endpoint given by:
#
#   JANUS_EXTERNAL_FALKOR=1        enable this mode
#   JANUS_ENGINE__HOST             DB host (default 127.0.0.1)
#   JANUS_ENGINE__PORT             DB port (default 6379)
#   JANUS_DAEMON__HTTP__PORT       daemon port (default 8765)
#   JANUS_SMOKE_DEAD_PORT          closed port used to simulate an outage
#
# WHY the "database is down" gates work differently here:
#   T6/T8/T9 need the daemon to lose its database. The original script does
#   that by killing the FalkorDB process. We do not own the external
#   database — and in CI it is a service container the runner manages, so
#   killing it would be rude to the runner and would break the container for
#   the rest of the job. Instead we point the daemon at a CLOSED PORT,
#   which produces exactly the same observable state the gates assert on:
#   the daemon cannot reach a FalkorDB, so probe() reports DEAD, the breaker
#   counts the failure, and /health answers 503 + degraded. The database
#   itself is never touched.
#
#   DEAD_PORT must differ from the live DB port and from the daemon's HTTP
#   port, otherwise the simulated outage is indistinguishable from the real
#   one. The script refuses to run if that is not true.
#
# Every assertion, the daemon lifecycle, and the log scraping are IDENTICAL
# in both modes — a gate that passes in external mode asserts the same thing
# it asserts in local mode.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DAEMON_PORT="${DAEMON_PORT:-8765}"
FALKOR_PID_FILE="${FALKOR_PID_FILE:-./data/falkordb/falkordb.pid}"
DAEMON_LOG="$(mktemp -t daemon_smoke_XXXXXX.log)"

DAEMON_PID=""


# Interpreter used for every python call in this script. It must be an
# interpreter that can import janus_graph AND graphiti_core — a bare
# system `python3` on a CI runner cannot (the dependencies live in the
# venv that `uv venv` created). Measured 2026-11-01: the first CI-shaped
# local run failed with
#   ModuleNotFoundError: No module named 'graphiti_core'
# and the daemon never started.
#
# The default stays `python3` so the existing local workflow is unchanged;
# CI passes PYTHON_BIN=.venv/bin/python.
PYTHON_BIN="${PYTHON_BIN:-python3}"

DAEMON_PID=""

# The daemon binds its HTTP port from JANUS_DAEMON__HTTP__PORT, while this
# script curls DAEMON_PORT. Two names for one setting is a trap: if they
# disagree, the gates silently assert against whatever daemon happens to
# listen on DAEMON_PORT — on a machine where the production daemon runs
# there, that means asserting against PRODUCTION. Measured 2026-11-01: the
# first external-mode run reported total_episodes=12734 and a sweep
# timestamp from 15 minutes earlier, i.e. the production daemon answered the
# probes. DAEMON_PORT is therefore the single source of truth and is pushed
# into the daemon's environment here.
export JANUS_DAEMON__HTTP__PORT="$DAEMON_PORT"

# External-DB mode: the database belongs to somebody else.
EXTERNAL_FALKOR="${JANUS_EXTERNAL_FALKOR:-0}"
FALKOR_HOST="${JANUS_ENGINE__HOST:-127.0.0.1}"
FALKOR_PORT="${JANUS_ENGINE__PORT:-6379}"
# A port nothing listens on — used to simulate "Falkor unreachable" without
# touching the real database.
DEAD_PORT="${JANUS_SMOKE_DEAD_PORT:-6399}"

# Colors for output
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

cleanup() {
    echo -e "
${YELLOW}[cleanup]${NC} stopping daemon"
    stop_daemon
    # Only stop falkordb if WE started it. In external mode the database
    # belongs to the caller and must be left alone.
    if [ "$EXTERNAL_FALKOR" != "1" ] && [ -f "$FALKOR_PID_FILE" ]; then
        pid=$(cat "$FALKOR_PID_FILE" 2>/dev/null || true)
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
        rm -f "$FALKOR_PID_FILE"
    fi
    rm -f "$DAEMON_LOG"
}
trap cleanup EXIT

# Stop the daemon and wait for it, but never longer than 10s — a daemon that
# ignores SIGTERM must not hang the whole job.
stop_daemon() {
    [ -n "$DAEMON_PID" ] || return 0
    kill "$DAEMON_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
        kill -0 "$DAEMON_PID" 2>/dev/null || break
        sleep 0.5
    done
    if kill -0 "$DAEMON_PID" 2>/dev/null; then
        echo -e "${YELLOW}[stop_daemon]${NC} daemon did not exit in 10s — SIGKILL"
        kill -9 "$DAEMON_PID" 2>/dev/null || true
    fi
    wait "$DAEMON_PID" 2>/dev/null || true
    DAEMON_PID=""
}

start_falkordb() {
    echo -e "${YELLOW}[start_falkordb]${NC}"
    "$PYTHON_BIN" -c "
from janus_graph.config import JanusSettings
from janus_graph.engine.server import FalkorDBServerManager
mgr = FalkorDBServerManager(JanusSettings().engine)
print('STARTED' if mgr.start() else 'FAILED')
"
}

ensure_falkordb_up() {
    if [ "$EXTERNAL_FALKOR" = "1" ]; then
        echo -e "${GREEN}[ensure_falkordb_up]${NC} external mode — using ${FALKOR_HOST}:${FALKOR_PORT} (not managed by this script)"
        return 0
    fi
    if [ -f "$FALKOR_PID_FILE" ]; then
        pid=$(cat "$FALKOR_PID_FILE" 2>/dev/null || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}[ensure_falkordb_up]${NC} already running (pid=$pid)"
            return 0
        fi
    fi
    start_falkordb
    sleep 1
}

# Fail fast, and with a useful message, when the external database is not
# actually there. Without this the T1 gate fails with a confusing "expected
# 200, got 503" instead of "the service container never came up".
preflight_external_db() {
    if [ "$DEAD_PORT" = "$FALKOR_PORT" ]; then
        echo -e "${RED}[preflight]${NC} DEAD_PORT ($DEAD_PORT) == DB port — the simulated outage would be a no-op"
        exit 1
    fi
    if [ "$DEAD_PORT" = "$DAEMON_PORT" ]; then
        echo -e "${RED}[preflight]${NC} DEAD_PORT ($DEAD_PORT) == daemon port — the simulated outage would be a no-op"
        exit 1
    fi
    echo -e "${YELLOW}[preflight]${NC} probing external FalkorDB at ${FALKOR_HOST}:${FALKOR_PORT}"
    if ! "$PYTHON_BIN" -c "
from janus_graph.config import JanusSettings
from janus_graph.daemon.falkordb_supervisor import DaemonSupervisor, ProbeState
import asyncio, sys
cfg = JanusSettings()
sup = DaemonSupervisor(cfg.engine, cfg.daemon.falkor_circuit)
state = asyncio.run(sup.probe_strict())
print('probe_strict =', state)
sys.exit(0 if state is ProbeState.ALIVE else 1)
"; then
        echo -e "${RED}[preflight]${NC} external FalkorDB NOT reachable at ${FALKOR_HOST}:${FALKOR_PORT}"
        exit 1
    fi
    echo -e "${GREEN}[preflight]${NC} external FalkorDB reachable"
}

# Refuse to run when something else already listens on the daemon port.
#
# The gates kill the daemon they started, but they cannot kill a daemon they
# did NOT start. If something is already on DAEMON_PORT, this script would
# (a) never start its own daemon, (b) assert against the other process, and
# (c) leave that process running. The 2026-11-01 measurement in the header
# comment is exactly this failure: the gates passed while the PRODUCTION
# daemon answered every probe. Failing loudly is the only safe answer.
preflight_daemon_port_free() {
    local busy
    # curl prints 000 itself when it cannot connect, so the fallback must not
    # add a second "000" — `busy` would become "000000" and never equal "000".
    # `|| true` is required: curl exits 7 on connection refused, and under
    # `set -e` a failing command substitution would abort the whole script
    # before the check could run.
    busy=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$DAEMON_PORT/health" 2>/dev/null) || true
    busy="${busy:-000}"
    if [ "$busy" != "000" ]; then
        echo -e "${RED}[preflight]${NC} port $DAEMON_PORT is already serving HTTP (status $busy)."
        echo -e "${RED}[preflight]${NC} refusing to run: the gates would assert against a daemon this script does not own."
        echo -e "${RED}[preflight]${NC} set DAEMON_PORT to a free port (e.g. DAEMON_PORT=8766)."
        exit 1
    fi
    echo -e "${GREEN}[preflight]${NC} port $DAEMON_PORT is free"
}

start_daemon() {
    echo -e "${YELLOW}[start_daemon]${NC}"
    # The daemon reads engine host/port and its own HTTP port from the
    # environment (JANUS_ prefix, '__' nested delimiter). Both modes use the
    # same mechanism, so the local and CI paths exercise the same config
    # plumbing.
    "$PYTHON_BIN" -m janus_graph.daemon > "$DAEMON_LOG" 2>&1 &
    DAEMON_PID=$!
    # Wait up to 10s for the daemon to answer /health at all.
    #
    # NOTE: "answer at all", not "answer 200". In external mode the daemon
    # is deliberately started against a closed port, and a daemon that cannot
    # reach its database answers 503 — which is a healthy daemon reporting a
    # degraded database. Waiting for 200 there would time out and fail the
    # gate for the wrong reason.
    for i in $(seq 1 20); do
        # `|| true`: curl exits 7 when the daemon is not listening yet, and
        # under `set -e` that would abort the script instead of retrying.
        # curl already prints "000" for a failed connection, so there is no
        # fallback echo here — adding one would yield "000000".
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$DAEMON_PORT/health" 2>/dev/null) || true
        code="${code:-000}"
        if [ "$code" != "000" ]; then
            echo -e "${GREEN}[start_daemon]${NC} ready (pid=$DAEMON_PID, /health → $code)"
            return 0
        fi
        sleep 0.5
    done
    echo -e "${RED}[start_daemon]${NC} failed to become ready — see $DAEMON_LOG"
    cat "$DAEMON_LOG"
    return 1
}

kill_falkordb() {
    if [ "$EXTERNAL_FALKOR" = "1" ]; then
        # We do not own the external database, so we do not kill it. The
        # "outage" is simulated by pointing the daemon at DEAD_PORT.
        return 0
    fi
    if [ -f "$FALKOR_PID_FILE" ]; then
        pid=$(cat "$FALKOR_PID_FILE" 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null || true
            sleep 0.5
        fi
        rm -f "$FALKOR_PID_FILE"
    fi
}

# Point the daemon at the closed port for the duration of the current gate.
# Restored by restore_falkordb_endpoint().
use_dead_endpoint() {
    echo -e "${YELLOW}[use_dead_endpoint]${NC} engine → ${FALKOR_HOST}:${DEAD_PORT} (simulated outage; database untouched)"
    export JANUS_ENGINE__PORT="$DEAD_PORT"
}

restore_falkordb_endpoint() {
    export JANUS_ENGINE__PORT="$FALKOR_PORT"
}

assert_status() {
    local expected="$1"; local actual="$2"; local label="$3"
    if [ "$actual" = "$expected" ]; then
        echo -e "${GREEN}[PASS]${NC} $label (status=$actual)"
        return 0
    else
        echo -e "${RED}[FAIL]${NC} $label — expected status=$expected, got $actual"
        return 1
    fi
}

# ─── T1: happy path ────────────────────────────────────────────────────
gate_t1() {
    echo -e "
${YELLOW}=== T1: happy path ===${NC}"
    ensure_falkordb_up
    start_daemon

    response=$(curl -s "http://127.0.0.1:$DAEMON_PORT/health" -w "
%{http_code}")
    status=$(echo "$response" | tail -1)
    body=$(echo "$response" | sed '$d')

    echo "body: $body"
    assert_status 200 "$status" "T1 /health returns 200"

    echo "$body" | "$PYTHON_BIN" -c "
import json, sys
d = json.loads(sys.stdin.read())
assert d['status'] == 'ready', f\"status={d['status']} (expected ready)\"
assert d['falkor_ok'] is True
assert d['circuit']['state'] == 'closed'
print('T1 body asserts: PASS')
"

    stop_daemon
}

# ─── T6: Falkor down → 503 ─────────────────────────────────────────────
gate_t6() {
    echo -e "
${YELLOW}=== T6: Falkor down → 503 ===${NC}"
    ensure_falkordb_up
    start_daemon

    if [ "$EXTERNAL_FALKOR" = "1" ]; then
        # First prove the database is reachable (a 200 from a daemon that has
        # its database), then take the database away by pointing a second
        # daemon instance at the closed port.
        healthy=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$DAEMON_PORT/health")
        if [ "$healthy" != "200" ]; then
            echo -e "${RED}[FAIL]${NC} T6 pre-check — daemon with a live database answered $healthy, expected 200"
            stop_daemon
            return 1
        fi
        stop_daemon
        use_dead_endpoint
        start_daemon
    else
        # Stop falkordb under the running daemon.
        kill_falkordb
    fi

    response=$(curl -s "http://127.0.0.1:$DAEMON_PORT/health" -w "
%{http_code}")
    status=$(echo "$response" | tail -1)
    body=$(echo "$response" | sed '$d')

    echo "body: $body"
    assert_status 503 "$status" "T6 /health returns 503"

    echo "$body" | "$PYTHON_BIN" -c "
import json, sys
d = json.loads(sys.stdin.read())
assert d['status'] == 'degraded'
assert d['falkor_ok'] is False
print('T6 body asserts: PASS')
"

    stop_daemon
    restore_falkordb_endpoint
}

# ─── T8: circuit-breaker ───────────────────────────────────────────────
gate_t8() {
    echo -e "
${YELLOW}=== T8: circuit-breaker ===${NC}"
    # Use short reset_timeout for test speed.
    export JANUS_DAEMON__FALKOR_CIRCUIT__RESET_TIMEOUT_SEC=5
    export JANUS_DAEMON__FALKOR_CIRCUIT__FAILURE_THRESHOLD=3
    ensure_falkordb_up

    if [ "$EXTERNAL_FALKOR" = "1" ]; then
        # Simulated outage: the daemon talks to a closed port for the whole
        # gate, so every /health below is a probe failure. Connecting to a
        # closed local port is refused immediately, so the failures land well
        # inside the 5s cooldown window.
        use_dead_endpoint
        start_daemon

        # Trip the circuit (threshold=3).
        for i in 1 2 3; do
            curl -s "http://127.0.0.1:$DAEMON_PORT/health" -o /dev/null
        done
    else
        start_daemon

        # Trip the circuit: 3 failures.
        for i in 1 2 3; do
            kill_falkordb
            curl -s "http://127.0.0.1:$DAEMON_PORT/health" -o /dev/null
            start_falkordb > /dev/null
            sleep 0.5
            kill_falkordb  # immediately down again
            curl -s "http://127.0.0.1:$DAEMON_PORT/health" -o /dev/null
        done
    fi

    # Wait a moment, then probe again.
    sleep 1

    # Within 5s reset window, probes should be SKIPPED.
    sleep 2
    skipped_count=$(grep -c "probe skipped" "$DAEMON_LOG" || true)
    echo -e "${YELLOW}[T8]${NC} probe-skipped log count: $skipped_count (expect ≥ 1)"

    response=$(curl -s "http://127.0.0.1:$DAEMON_PORT/health" -w "
%{http_code}")
    body=$(echo "$response" | sed '$d')
    echo "T8 body: $body"

    echo "$body" | "$PYTHON_BIN" -c "
import json, sys
d = json.loads(sys.stdin.read())
assert d['circuit']['state'] == 'open', f\"state={d['circuit']['state']} (expected open)\"
assert d['circuit']['failure_count'] >= 3
assert d['circuit']['retry_at'] is not None
print('T8 body asserts: PASS')
"

    if [ "$skipped_count" -lt 1 ]; then
        echo -e "${RED}[FAIL]${NC} T8 expected probe-skipped logs during cooldown"
        stop_daemon
        restore_falkordb_endpoint
        unset JANUS_DAEMON__FALKOR_CIRCUIT__RESET_TIMEOUT_SEC
        unset JANUS_DAEMON__FALKOR_CIRCUIT__FAILURE_THRESHOLD
        return 1
    fi
    echo -e "${GREEN}[PASS]${NC} T8 probe-skipped logs observed"

    stop_daemon
    restore_falkordb_endpoint
    unset JANUS_DAEMON__FALKOR_CIRCUIT__RESET_TIMEOUT_SEC
    unset JANUS_DAEMON__FALKOR_CIRCUIT__FAILURE_THRESHOLD
}

# ─── T9: OPEN doesn't break T6/T7 ──────────────────────────────────────
gate_t9() {
    echo -e "
${YELLOW}=== T9: circuit OPEN preserves degraded body ===${NC}"
    export JANUS_DAEMON__FALKOR_CIRCUIT__RESET_TIMEOUT_SEC=10
    export JANUS_DAEMON__FALKOR_CIRCUIT__FAILURE_THRESHOLD=2
    ensure_falkordb_up

    if [ "$EXTERNAL_FALKOR" = "1" ]; then
        use_dead_endpoint
        start_daemon

        # Trip the circuit quickly (threshold=2).
        curl -s "http://127.0.0.1:$DAEMON_PORT/health" -o /dev/null
        curl -s "http://127.0.0.1:$DAEMON_PORT/health" -o /dev/null
    else
        start_daemon

        # Trip the circuit quickly (threshold=2).
        kill_falkordb
        curl -s "http://127.0.0.1:$DAEMON_PORT/health" -o /dev/null
        curl -s "http://127.0.0.1:$DAEMON_PORT/health" -o /dev/null
    fi

    # /health must still return 503 with full body shape while OPEN.
    response=$(curl -s "http://127.0.0.1:$DAEMON_PORT/health" -w "
%{http_code}")
    status=$(echo "$response" | tail -1)
    body=$(echo "$response" | sed '$d')

    assert_status 503 "$status" "T9 /health 503 (OPEN state)"
    echo "$body" | "$PYTHON_BIN" -c "
import json, sys
d = json.loads(sys.stdin.read())
assert 'daemon_version' in d
assert 'circuit' in d
assert d['falkor_ok'] is False
print('T9 body shape preserved: PASS')
"

    stop_daemon
    restore_falkordb_endpoint
    unset JANUS_DAEMON__FALKOR_CIRCUIT__RESET_TIMEOUT_SEC
    unset JANUS_DAEMON__FALKOR_CIRCUIT__FAILURE_THRESHOLD
}

# ─── main ─────────────────────────────────────────────────────────────
gates="${1:-all}"

if [ "$EXTERNAL_FALKOR" = "1" ]; then
    preflight_external_db
fi
preflight_daemon_port_free

case "$gates" in
    t1) gate_t1 ;;
    t6) gate_t6 ;;
    t8) gate_t8 ;;
    t9) gate_t9 ;;
    all)
        gate_t1
        gate_t6
        gate_t8
        gate_t9
        echo -e "
${GREEN}=== ALL GATES PASSED ===${NC}"
        ;;
    *) echo "Usage: $0 [t1|t6|t8|t9|all]"; exit 1 ;;
esac
