# Auto-Restart Supervisor (push trigger)

> **Status**: shipped in `v0.7.0-restart-supervisor` (Plan #3 Phase 1–3)
> **Module**: `janus_graph.daemon.falkordb_restart_policy.FalkorRestartPolicy`
> **Hook point**: `janus_graph.daemon.falkordb_supervisor.FalkorDBServerManager` circuit OPEN event

## 1. Why this exists

FalkorDB can crash silently (segfault, OOM-kill, port collision) without any
process supervisor. Before this feature, a crashed engine meant:

1. `record_failure()` increments circuit counter on every probe
2. After 3 failures circuit goes OPEN → `/health` returns 503 forever
3. Operator has to manually `pmc restart janus-graph-daemon` (also restarts
   the daemon, drops in-flight MCP requests, costs ~30s downtime)

The auto-restart supervisor turns step 3 into a self-healing action: the
daemon detects `circuit OPEN`, calls `manager.start()` on its own, and
probes back to CLOSED — **zero operator intervention, zero daemon restart**.

## 2. State machine

```
                          circuit OPEN event
                                 │
                                 ▼
            ┌────────────────────────────────────┐
            │ FalkorRestartPolicy.attempt_       │
            │ restart()                          │
            └────────────────────────────────────┘
                                 │
                ┌────────────────┼─────────────────┐
                ▼                ▼                 ▼
        operator_disabled  FATAL_DEGRADED    cooldown active
        → skip + log       → skip + ERROR    → skip + INFO
                │                │                 │
                └────────────────┴─────────────────┘
                                 │
                          check passes
                                 ▼
                    manager.start() runs
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
            start=True                    start=False/raises
            → record success              → record failure
            → /health restart_policy      → rolling-window budget
              shows current state           tracks; after max_per_hour
                                            → FATAL_DEGRADED
```

The policy is a **single instance per daemon process**. It is wired at
`lifespan.py` boot from `DaemonSettings.falkor_restart_policy` and
attached to the `FalkorDBServerManager` via a callback registration.

## 3. Tunables (`falkor_restart_policy`)

| Field          | Default | Env var                                        | Meaning                                      |
| -------------- | ------- | ---------------------------------------------- | -------------------------------------------- |
| `cooldown_sec` | 60      | `JANUS_DAEMON__FALKOR_RESTART_POLICY__COOLDOWN_SEC` | Minimum gap between consecutive restarts. Prevents thrash. |
| `max_per_hour` | 5       | `JANUS_DAEMON__FALKOR_RESTART_POLICY__MAX_PER_HOUR` | Rolling 1-hour budget. After N restarts in any 1-hour window → FATAL_DEGRADED (operator intervention required). |

**Precedence**: env > `config.yaml` / `config.example.yaml` > builtin default.

## 4. /health snapshot

Phase 2 added a `restart_policy` key to `GET /health` (additive — T9
invariant preserved: old clients still see `circuit` + `queue_stats` +
`falkor_ok` keys unchanged):

```json
{
  "status": "degraded",
  "falkor_ok": false,
  "circuit": {"state": "open", "failure_count": 15, ...},
  "restart_policy": {
    "state": "ACTIVE",
    "restarts_in_window": 2,
    "max_per_hour": 5,
    "seconds_since_last_restart": 12.3,
    "cooldown_sec": 60,
    "disabled_reason": null
  },
  "daemon_version": "0.5.0-restart-supervisor",
  "queue_stats": {...}
}
```

`restart_policy.state` values:

- `ACTIVE` — policy armed, eligible to restart on next OPEN event
- `COOLING_DOWN` — recent restart, waiting for `cooldown_sec`
- `BUDGET_EXHAUSTED` — `max_per_hour` reached in rolling window
- `FATAL_DEGRADED` — operator intervention required (manual `pmc restart`)
- `OPERATOR_DISABLED` — `policy.disable(reason="...")` was called (e.g. falkordb.so binary missing)

## 5. Operator runbook

| Symptom                                       | Cause                          | Action                                                                |
| --------------------------------------------- | ------------------------------ | --------------------------------------------------------------------- |
| `/health.restart_policy.state == "FATAL_DEGRADED"` | Max restarts exhausted in 1h window | Check Falkor logs (e.g. `data/logs/falkordb.log`); fix root cause; `pmc restart janus-graph-daemon` to clear state |
| `/health.restart_policy.state == "OPERATOR_DISABLED"` | falkordb.so binary missing OR explicit `policy.disable()` call | Re-add binary to `bin/`; or `policy.enable()` after manual intervention |
| Daemon logs `restart_policy: manager.start() returned False` | Spawn raced with existing process | Wait one probe cycle; if persists, check port 6379 (`ss -tlnp \| grep 6379`) for orphan redis/falkor process |
| `/health` 503 but `restart_policy.state == "ACTIVE"` and `falkor_ok == false` | Restart attempted but probe still failing | Check `data/logs/janus_graph.log` for the restart report — `log_tail_lines_on_fail=20` is attached |

## 6. Edge cases handled

| Case | Behavior |
| ---- | -------- |
| `falkordb.so` binary missing at boot | `lifespan._check_falkordb_binary()` calls `policy.disable("binary_missing")` — operator gets clear signal in /health |
| Port 6379 held by another process | `manager.start()` returns False → restart counts as failure → enters cooldown |
| Concurrent `record_failure()` from cron_loop + MCP + health probe | `threading.Lock` around `_attempt_restart_locked()` — only 1 spawn per failure event (verified by `test_concurrent_record_failure_during_restart_blocks`) |
| Falkor starts but immediately crashes | `_ping()` after `start()` returns False → recorded as failure; rolling budget tracks |
| Daemon SIGTERM mid-restart | In-flight `manager.start()` runs to completion (~2s budget); new `record_failure()` calls during shutdown are no-ops |
| Manual falkordb already running on port 6379 | `manager.start()` returns False (port already bound) → counts as failure → no double-spawn risk |

## 7. Tests

- `tests/test_falkor_circuit.py` — 10 tests (Phase 0 baseline, unchanged)
- `tests/test_falkordb_restart_policy.py` — 11 tests (Phase 1 unit)
- `tests/test_falkordb_supervisor_integration.py` — 6 tests (Phase 2 integration)
- **Total: 27/27 PASS in 0.14s**

Sandbox verify (port 6399, real `falkordb.so` v4.x):

```
Step 4: Auto-restart via FalkorDBServerManager.start()
        restart: True
        restart elapsed: 1036ms (target < 2000ms) ✅
Step 5: PING after auto-restart → +PONG
```

## 8. References

- Plan: `janus-graph/falkordb-auto-restart-supervisor-20260909.md`
- Phase 2 commit: `b3823ac` (concurrent-safe restart + binary boot check + /health snapshot)
- Phase 3 commit: `v0.7.0-restart-supervisor` (config schema + this doc + README bullet)
- Phase 1 commit: `9c64ada` (initial FalkorRestartPolicy module)
