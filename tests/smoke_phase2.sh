#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Phase 2 integration smoke gates (Plan #2 §6 verification).
#
# Gates:
#   T1 — happy path: Falkor alive → GET /health returns 200 + status=ready
#   T6 — Falkor down: pkill → GET /health returns 503 + status=degraded
#   T8 — circuit-breaker: 3 kills in 60s → circuit_open, 0 probe log in cooldown
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
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DAEMON_PORT="${DAEMON_PORT:-8765}"
FALKOR_PID_FILE="${FALKOR_PID_FILE:-./data/falkordb/falkordb.pid}"
DAEMON_LOG="$(mktemp -t daemon_smoke_XXXXXX.log)"
DAEMON_PID=""

# Colors for output
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

cleanup() {
    echo -e "
${YELLOW}[cleanup]${NC} stopping daemon + falkordb"
    [ -n "$DAEMON_PID" ] && kill "$DAEMON_PID" 2>/dev/null || true
    # If we started falkordb ourselves, stop it.
    if [ -f "$FALKOR_PID_FILE" ]; then
        pid=$(cat "$FALKOR_PID_FILE" 2>/dev/null || true)
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
        rm -f "$FALKOR_PID_FILE"
    fi
    rm -f "$DAEMON_LOG"
}
trap cleanup EXIT

start_falkordb() {
    echo -e "${YELLOW}[start_falkordb]${NC}"
    python3 -c "
from janus_graph.config import JanusSettings
from janus_graph.engine.server import FalkorDBServerManager
mgr = FalkorDBServerManager(JanusSettings().engine)
print('STARTED' if mgr.start() else 'FAILED')
"
}

ensure_falkordb_up() {
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

start_daemon() {
    echo -e "${YELLOW}[start_daemon]${NC}"
    python3 -m janus_graph.daemon > "$DAEMON_LOG" 2>&1 &
    DAEMON_PID=$!
    # Wait up to 10s for /health to respond.
    for i in $(seq 1 20); do
        if curl -sf "http://127.0.0.1:$DAEMON_PORT/health" -o /dev/null 2>&1; then
            echo -e "${GREEN}[start_daemon]${NC} ready (pid=$DAEMON_PID)"
            return 0
        fi
        sleep 0.5
    done
    echo -e "${RED}[start_daemon]${NC} failed to become ready — see $DAEMON_LOG"
    cat "$DAEMON_LOG"
    return 1
}

kill_falkordb() {
    if [ -f "$FALKOR_PID_FILE" ]; then
        pid=$(cat "$FALKOR_PID_FILE" 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null || true
            sleep 0.5
        fi
        rm -f "$FALKOR_PID_FILE"
    fi
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

    echo "$body" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
assert d['status'] == 'ready', f\"status={d['status']} (expected ready)\"
assert d['falkor_ok'] is True
assert d['circuit']['state'] == 'closed'
print('T1 body asserts: PASS')
"

    kill "$DAEMON_PID"
    wait "$DAEMON_PID" 2>/dev/null || true
    DAEMON_PID=""
}

# ─── T6: Falkor down → 503 ─────────────────────────────────────────────
gate_t6() {
    echo -e "
${YELLOW}=== T6: Falkor down → 503 ===${NC}"
    ensure_falkordb_up
    start_daemon

    # Stop falkordb.
    kill_falkordb

    response=$(curl -s "http://127.0.0.1:$DAEMON_PORT/health" -w "
%{http_code}")
    status=$(echo "$response" | tail -1)
    body=$(echo "$response" | sed '$d')

    echo "body: $body"
    assert_status 503 "$status" "T6 /health returns 503"

    echo "$body" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
assert d['status'] == 'degraded'
assert d['falkor_ok'] is False
print('T6 body asserts: PASS')
"

    kill "$DAEMON_PID"
    wait "$DAEMON_PID" 2>/dev/null || true
    DAEMON_PID=""
}

# ─── T8: circuit-breaker ───────────────────────────────────────────────
gate_t8() {
    echo -e "
${YELLOW}=== T8: circuit-breaker ===${NC}"
    # Use short reset_timeout for test speed.
    export JANUS_DAEMON__FALKOR_CIRCUIT__RESET_TIMEOUT_SEC=5
    export JANUS_DAEMON__FALKOR_CIRCUIT__FAILURE_THRESHOLD=3
    ensure_falkordb_up
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

    echo "$body" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
assert d['circuit']['state'] == 'open', f\"state={d['circuit']['state']} (expected open)\"
assert d['circuit']['failure_count'] >= 3
assert d['circuit']['retry_at'] is not None
print('T8 body asserts: PASS')
"

    if [ "$skipped_count" -lt 1 ]; then
        echo -e "${RED}[FAIL]${NC} T8 expected probe-skipped logs during cooldown"
        return 1
    fi
    echo -e "${GREEN}[PASS]${NC} T8 probe-skipped logs observed"

    kill "$DAEMON_PID"
    wait "$DAEMON_PID" 2>/dev/null || true
    DAEMON_PID=""
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
    start_daemon

    # Trip the circuit quickly (threshold=2).
    kill_falkordb
    curl -s "http://127.0.0.1:$DAEMON_PORT/health" -o /dev/null
    curl -s "http://127.0.0.1:$DAEMON_PORT/health" -o /dev/null

    # /health must still return 503 with full body shape while OPEN.
    response=$(curl -s "http://127.0.0.1:$DAEMON_PORT/health" -w "
%{http_code}")
    status=$(echo "$response" | tail -1)
    body=$(echo "$response" | sed '$d')

    assert_status 503 "$status" "T9 /health 503 (OPEN state)"
    echo "$body" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
assert 'daemon_version' in d
assert 'circuit' in d
assert d['falkor_ok'] is False
print('T9 body shape preserved: PASS')
"

    kill "$DAEMON_PID"
    wait "$DAEMON_PID" 2>/dev/null || true
    DAEMON_PID=""
    unset JANUS_DAEMON__FALKOR_CIRCUIT__RESET_TIMEOUT_SEC
    unset JANUS_DAEMON__FALKOR_CIRCUIT__FAILURE_THRESHOLD
}

# ─── main ─────────────────────────────────────────────────────────────
gates="${1:-all}"
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
