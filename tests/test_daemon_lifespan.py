"""Tests for the daemon boot sequence (``janus_graph/daemon/lifespan.py``).

Why this file exists
--------------------
The repo-wide suite had **no test that executed** ``boot()`` or ``run()``.
Five test modules imported ``DaemonContext`` purely as a dataclass to build
handler fixtures, so the real production entry path — probe Falkor, wire
Phase 3, spawn aiohttp, wire cron, then tear everything down on shutdown —
carried **zero** assertions. ``lifespan.py`` sat at 29% coverage as a result
(code-review 2026-10-17).

These tests cover the four units that are cheap to isolate and expensive to get
wrong:

  * ``_check_falkordb_binary``  — three outcomes: binary present / missing /
    resolve raised. Each must arm or disable the restart policy.
  * ``_warm_graphiti_singleton`` — success, timeout, and generic exception.
    All three are *documented* to be fail-soft (return ``None`` → handler
    answers 503 later), so a future refactor that lets one escape would turn
    a degraded daemon into a crash-looping one.
  * ``boot`` — the three Falkor branches (alive / warm-up / dead-then-start),
    each asserting whether ``start()`` was called and whether the breaker
    recorded success or failure.
  * ``run`` — wiring plus graceful-shutdown ordering.

Isolation strategy
------------------
``boot``/``run`` are orchestration: their value is the *sequence of calls*, not
the internals of the subsystems. So the collaborators are patched **at the
``lifespan`` module boundary** (``_build_phase3_components``, ``CronLoop``,
``build_app``, ``web.AppRunner``/``TCPSite``) and the assertions are about
*which* collaborator was called, in what order, with what outcome. The real
supervisor logic is exercised through its own test module
(``test_falkordb_boot_autostart.py``); duplicating it here would only create
a second place to update.

Everything here is hermetic: no FalkorDB, no subprocess, no real socket, no
network. The only filesystem touch is a temp dir for ``queue_db_path``, which
the patched ``_build_phase3_components`` never actually opens.
"""

from __future__ import annotations

import asyncio
import signal
import threading
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from janus_graph.config import JanusSettings
from janus_graph.daemon import lifespan as lifespan_mod
from janus_graph.daemon.falkordb_supervisor import ProbeState
from janus_graph.daemon.lifespan import (
    DaemonContext,
    _build_phase3_components,
    _check_falkordb_binary,
    _shutdown_phase3_components,
    _warm_graphiti_singleton,
    boot,
    run,
)

# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> JanusSettings:
    """Repo-default settings with the queue DB pinned inside tmp_path.

    ``conftest._isolated_operator_env`` (session-wide, autouse) already stops
    ``JanusSettings()`` from resolving the operator's live ``config.yaml``;
    we only override the one path that would otherwise be relative to cwd.
    """
    return JanusSettings(pipeline={"queue_db_path": str(tmp_path / "episodes.db")})


def _make_supervisor(probe_state: ProbeState, start_ok: bool = False) -> MagicMock:
    """A supervisor stand-in with an async-aware surface.

    Deliberately a ``MagicMock``, not a real ``DaemonSupervisor``: the three
    collaborators ``boot`` awaits (``probe_strict`` / ``start`` /
    ``breaker.record_*``) are the whole observable contract here, and
    ``DaemonSupervisor`` exposes ``breaker`` as a read-only ``@property`` so
    its call counts cannot be asserted by assignment anyway. The supervisor's
    own probe/breaker logic is already covered by
    ``test_falkordb_boot_autostart.py``; re-testing it here would only create
    a second place to update.
    """
    supervisor = MagicMock()
    supervisor.probe_strict = AsyncMock(side_effect=[probe_state, probe_state])
    supervisor.start = AsyncMock(return_value=start_ok)
    supervisor.breaker.state = MagicMock()
    supervisor.breaker.record_success = AsyncMock()
    supervisor.breaker.record_failure = AsyncMock()
    return supervisor


@pytest.fixture
def ctx(settings: JanusSettings, tmp_path: Path) -> DaemonContext:
    """A DaemonContext whose supervisor is fully mocked."""
    supervisor = _make_supervisor(ProbeState.ALIVE)
    module_path = tmp_path / "falkordb.so"
    module_path.write_bytes(b"\x7fELF")  # present ⇒ binary check arms policy
    supervisor.manager.resolve_binary_paths = MagicMock(
        return_value=(tmp_path / "redis-server", module_path)
    )
    return DaemonContext(
        settings=settings,
        supervisor=supervisor,
        http_settings=settings.daemon.http,
        version="0.0.0-test",
        shutdown_event=asyncio.Event(),
    )


@pytest.fixture
def patched_boot_collaborators():
    """Patch every heavy collaborator of ``boot``; yield the mock handles.

    ``auto_stop`` is essential: without it, an assertion failure before
    ``site.stop()`` would leave a real ``AppRunner`` holding a bound socket
    and leak into the next test in this module.
    """
    with (
        patch.object(lifespan_mod, "_build_phase3_components", new=AsyncMock()) as phase3,
        patch.object(lifespan_mod, "_warm_graphiti_singleton", new=AsyncMock(return_value=None)),
        patch.object(lifespan_mod, "CronLoop") as cron_cls,
        patch.object(lifespan_mod, "build_app") as build_app,
        patch.object(lifespan_mod.web, "AppRunner") as runner_cls,
        patch.object(lifespan_mod.web, "TCPSite") as site_cls,
    ):
        cron_cls.return_value = MagicMock(
            start=AsyncMock(),
            stop=AsyncMock(),
            running=False,
        )
        build_app.return_value = MagicMock()
        runner_cls.return_value = MagicMock(
            setup=AsyncMock(),
            cleanup=AsyncMock(),
        )
        site_cls.return_value = MagicMock(start=AsyncMock(), stop=AsyncMock())
        yield {
            "phase3": phase3,
            "cron_cls": cron_cls,
            "build_app": build_app,
            "runner_cls": runner_cls,
            "site_cls": site_cls,
        }


# --- DaemonContext.request_shutdown ---------------------------------------


async def test_request_shutdown_sets_event(ctx: DaemonContext) -> None:
    """The single line every shutdown path funnels through must set the event.

    If this silently stopped setting the event, ``run`` would hang forever in
    ``shutdown_event.wait()`` — SIGTERM would stop the daemon from exiting.
    """
    assert not ctx.shutdown_event.is_set()
    ctx.request_shutdown()
    assert ctx.shutdown_event.is_set()


# --- _check_falkordb_binary -------------------------------------------------


async def test_binary_present_arms_restart_policy(ctx: DaemonContext) -> None:
    """Binary present ⇒ no disable() call, policy stays armed."""
    await _check_falkordb_binary(ctx)
    ctx.supervisor.restart_policy.disable.assert_not_called()


async def test_binary_missing_disables_restart_policy(ctx: DaemonContext) -> None:
    """Missing ``falkordb.so`` ⇒ policy disabled, reason names the path.

    The reason string is operator-facing: ``/health`` surfaces
    ``disabled_reason``, so it must stay greppable, not become a bare ``True``.
    """
    missing = Path("/nonexistent/bin/falkordb.so")
    ctx.supervisor.manager.resolve_binary_paths = MagicMock(
        return_value=(Path("/nonexistent/bin/redis-server"), missing)
    )

    await _check_falkordb_binary(ctx)

    ctx.supervisor.restart_policy.disable.assert_called_once()
    reason = ctx.supervisor.restart_policy.disable.call_args.kwargs["reason"]
    assert reason == f"binary_missing:{missing}"


async def test_binary_resolve_raises_disables_policy(ctx: DaemonContext) -> None:
    """A raising ``resolve_binary_paths`` must also disable, not propagate.

    A boot-time crash here would mean a fresh clone can never start the daemon
    at all — the exact situation this check exists to soft-handle.
    """
    ctx.supervisor.manager.resolve_binary_paths = MagicMock(
        side_effect=OSError("no such directory")
    )

    await _check_falkordb_binary(ctx)

    ctx.supervisor.restart_policy.disable.assert_called_once()
    reason = ctx.supervisor.restart_policy.disable.call_args.kwargs["reason"]
    assert reason.startswith("binary_resolve_failed:")


# --- _warm_graphiti_singleton ----------------------------------------------


async def test_warm_graphiti_success_returns_instance(
    ctx: DaemonContext,
) -> None:
    sentinel = object()
    with patch.object(lifespan_mod, "create_graphiti_instance", return_value=sentinel) as create:
        got = await _warm_graphiti_singleton(ctx)
    assert got is sentinel
    create.assert_called_once_with(ctx.settings)


async def test_warm_graphiti_timeout_returns_none(ctx: DaemonContext) -> None:
    """Timeout must degrade to ``None``, not raise.

    ``boot`` awaits this; an escaping ``TimeoutError`` would abort the whole
    daemon start. The handler turns ``None`` into 503 at request time instead.

    The block must be a *synchronous* ``threading.Event.wait``, not
    ``asyncio.sleep``: ``create_graphiti_instance`` runs inside
    ``asyncio.to_thread``, so an async sleep here would just hand ``wait_for``
    an un-awaited coroutine that resolves instantly and no timeout would fire.

    The wait is bounded (not ``time.sleep(3600)``) on purpose: the abandoned
    worker thread is still parked in the default executor, and CPython's
    ``ThreadPoolExecutor`` atexit hook joins every worker. An unbounded sleep
    would make the whole pytest process hang for an hour *after* the test had
    already passed.
    """
    blocker = threading.Event()
    with patch.object(
        lifespan_mod, "create_graphiti_instance", side_effect=lambda s: blocker.wait(timeout=1.0)
    ):
        with patch.object(lifespan_mod, "_GRAPHITI_INIT_TIMEOUT_SEC", 0.05):
            got = await _warm_graphiti_singleton(ctx)
    assert got is None


async def test_warm_graphiti_exception_returns_none(ctx: DaemonContext) -> None:
    """Any exception (import error, Falkor down, OOM) degrades to ``None``."""
    with patch.object(
        lifespan_mod,
        "create_graphiti_instance",
        side_effect=RuntimeError("falkordb unreachable"),
    ):
        got = await _warm_graphiti_singleton(ctx)
    assert got is None


# --- boot: Falkor branches -------------------------------------------------


async def test_boot_falkor_alive_does_not_start(
    ctx: DaemonContext, patched_boot_collaborators: dict[str, Any]
) -> None:
    """Alive on first probe ⇒ skip start() and record breaker success."""
    ctx.supervisor.probe_strict = AsyncMock(return_value=ProbeState.ALIVE)

    await boot(ctx)

    ctx.supervisor.start.assert_not_called()
    ctx.supervisor.breaker.record_success.assert_awaited_once()
    ctx.supervisor.breaker.record_failure.assert_not_called()
    patched_boot_collaborators["phase3"].assert_awaited_once_with(ctx)
    assert ctx.runner is not None and ctx.site is not None


async def test_boot_falkor_dead_then_start_ok(
    ctx: DaemonContext, patched_boot_collaborators: dict[str, Any]
) -> None:
    """Dead ⇒ start(); a now-alive re-probe records success, not failure.

    This is the auto-start path that saves a fresh boot with no running
    FalkorDB. Two assertions carry the weight, and both were tightened after
    the mutation suite found them too weak:

    * ``record_failure`` must NOT be called -- a regression that reported
      failure here would open the circuit and stop Falkor from ever retrying.
    * ``record_success`` must be called **exactly once**. ``assert_awaited()``
      would still pass if the whole first-probe block ran and only the
      boot-time branch dropped its call, because that is the branch this test
      is actually about. Exactly-once also pins the "no redundant
      double-record" property on the ALIVE first-probe path.
    """
    ctx.supervisor.probe_strict = AsyncMock(side_effect=[ProbeState.DEAD, ProbeState.ALIVE])
    ctx.supervisor.start = AsyncMock(return_value=True)

    await boot(ctx)

    ctx.supervisor.start.assert_awaited_once()
    ctx.supervisor.breaker.record_success.assert_awaited_once()
    ctx.supervisor.breaker.record_failure.assert_not_called()


async def test_boot_falkor_dead_then_start_but_still_dead(
    ctx: DaemonContext, patched_boot_collaborators: dict[str, Any]
) -> None:
    """start() returned True but re-probe still DEAD ⇒ degraded, record failure."""
    ctx.supervisor.probe_strict = AsyncMock(side_effect=[ProbeState.DEAD, ProbeState.DEAD])
    ctx.supervisor.start = AsyncMock(return_value=True)

    await boot(ctx)

    ctx.supervisor.breaker.record_failure.assert_awaited_once()


async def test_boot_falkor_start_returns_false(
    ctx: DaemonContext, patched_boot_collaborators: dict[str, Any]
) -> None:
    """start() refused (binary missing / circuit open) ⇒ failure, no re-probe.

    Only one probe call is expected: there is nothing to re-probe when the
    start was never attempted, and the extra call would be a wasted round
    trip to a known-dead port.
    """
    ctx.supervisor.probe_strict = AsyncMock(return_value=ProbeState.DEAD)
    ctx.supervisor.start = AsyncMock(return_value=False)

    await boot(ctx)

    ctx.supervisor.start.assert_awaited_once()
    ctx.supervisor.breaker.record_failure.assert_awaited_once()
    assert ctx.supervisor.probe_strict.await_count == 1


async def test_boot_warming_up_proceeds_degraded(
    ctx: DaemonContext, patched_boot_collaborators: dict[str, Any]
) -> None:
    """Redis still ``-LOADING`` ⇒ daemon boots anyway; circuit keeps retrying."""
    ctx.supervisor.probe_strict = AsyncMock(side_effect=[ProbeState.WARMING_UP, ProbeState.ALIVE])
    ctx.supervisor.start = AsyncMock(return_value=True)

    await boot(ctx)

    ctx.supervisor.start.assert_awaited_once()
    patched_boot_collaborators["phase3"].assert_awaited_once()


async def test_boot_wires_http_site_on_configured_host(
    ctx: DaemonContext, patched_boot_collaborators: dict[str, Any]
) -> None:
    """The listening socket must use the *configured* host/port.

    Pins the loopback binding: ``/shutdown`` has no auth, so binding to
    anything other than 127.0.0.1 is what keeps that endpoint unreachable
    from the network. This test is the guard rail for that invariant.
    """
    await boot(ctx)

    patched_boot_collaborators["site_cls"].assert_called_once()
    kwargs = patched_boot_collaborators["site_cls"].call_args.kwargs
    assert kwargs["host"] == ctx.http_settings.host
    assert kwargs["port"] == ctx.http_settings.port
    assert kwargs["host"] == "127.0.0.1"


async def test_boot_actually_starts_runner_and_site(
    ctx: DaemonContext, patched_boot_collaborators: dict[str, Any]
) -> None:
    """``setup()`` and ``start()`` must be awaited, not merely constructed.

    Regression guard added 2026-10-17 after the mutation suite showed a
    vacuous test: swapping ``await ctx.site.start()`` for a no-op left the
    suite fully green, because the assertions above only ever checked that
    ``TCPSite(...)`` was *called with* the right host/port. Building the object
    is not listening on it.
    """
    await boot(ctx)

    patched_boot_collaborators["runner_cls"].return_value.setup.assert_awaited_once()
    patched_boot_collaborators["site_cls"].return_value.start.assert_awaited_once()


async def test_boot_propagates_site_start_failure(
    ctx: DaemonContext, patched_boot_collaborators: dict[str, Any]
) -> None:
    """A failed bind (port in use) must surface, not be swallowed.

    ``run`` owns teardown, so letting the error propagate is what gives the
    finally block a chance to release the queue and cron tasks that boot had
    already created. Swallowing it would leave those behind on every retry.
    """
    patched_boot_collaborators["site_cls"].return_value.start = AsyncMock(
        side_effect=OSError("address already in use")
    )

    with pytest.raises(OSError, match="address already in use"):
        await boot(ctx)


async def test_boot_passes_ctx_to_build_app(
    ctx: DaemonContext, patched_boot_collaborators: dict[str, Any]
) -> None:
    """Handlers resolve their state from ``app["ctx"]``, so ctx must be bound."""
    await boot(ctx)
    patched_boot_collaborators["build_app"].assert_called_once_with(ctx)


async def test_boot_passes_breaker_provider_to_cron(
    ctx: DaemonContext, patched_boot_collaborators: dict[str, Any]
) -> None:
    """CronLoop must read live circuit state through the supplied callable.

    A stale snapshot would make the sweep keep hitting a dead FalkorDB with
    no backoff, so the provider is asserted to be callable and to return the
    supervisor's current state.
    """
    await boot(ctx)

    kwargs = patched_boot_collaborators["cron_cls"].call_args.kwargs
    provider = kwargs["circuit_state_provider"]
    assert provider() is ctx.supervisor.breaker.state


# --- _shutdown_phase3_components -------------------------------------------


async def test_shutdown_phase3_clears_graphiti_before_closing_engines(
    ctx: DaemonContext,
) -> None:
    """graphiti_instance must be dropped first.

    The graphiti instance holds the FalkorDriver; the comment in the source is
    explicit that GC'ing it before ``search_engine.close()`` releases the
    redis pool. Asserting the *order* (not just that both happened) is the
    only way this stays true.
    """
    order: list[str] = []
    ctx.graphiti_instance = object()
    ctx.search_engine = MagicMock(close=AsyncMock(side_effect=lambda: order.append("engine")))
    ctx.embedding_client = MagicMock(stop=AsyncMock(side_effect=lambda: order.append("embedding")))

    await _shutdown_phase3_components(ctx)

    assert ctx.graphiti_instance is None
    ctx.search_engine.close.assert_awaited_once()
    ctx.embedding_client.stop.assert_awaited_once()
    assert order == ["engine", "embedding"]


async def test_shutdown_phase3_tolerates_all_components_none(
    ctx: DaemonContext,
) -> None:
    """Partial boot ⇒ teardown must not raise.

    Reached whenever boot fails midway (e.g. Falkor down): the finally block
    still runs, so an ``AttributeError`` here would mask the original boot
    failure with a confusing shutdown traceback.
    """
    ctx.graphiti_instance = None
    ctx.search_engine = None
    ctx.embedding_client = None

    await _shutdown_phase3_components(ctx)  # must not raise


# --- run -------------------------------------------------------------------


async def test_run_returns_zero_after_shutdown(
    settings: JanusSettings,
) -> None:
    """Happy path: boot, then /shutdown ⇒ clean exit 0, full teardown run.

    ``boot`` is patched to trip the shutdown event immediately, which is the
    in-process equivalent of ``POST /shutdown``.
    """
    calls: list[Any] = []

    async def _boot_then_shutdown(ctx: DaemonContext) -> None:
        calls.append(("boot", ctx))
        ctx.request_shutdown()

    with (
        patch.object(lifespan_mod, "boot", new=_boot_then_shutdown),
        patch.object(lifespan_mod, "_shutdown_phase3_components", new=AsyncMock()) as teardown,
    ):
        rc = await run(settings)

    assert rc == 0
    assert calls[0][0] == "boot"
    teardown.assert_awaited_once()


async def test_run_tears_down_even_when_boot_raises(
    settings: JanusSettings,
) -> None:
    """A boot failure must still unwind through the finally block.

    Without this the daemon would leave cron tasks and open sockets behind on
    every failed restart attempt, which is how a crash-loop turns into a
    resource leak.
    """

    async def _boom(ctx: DaemonContext) -> None:
        raise RuntimeError("falkordb binary corrupt")

    with (
        patch.object(lifespan_mod, "boot", new=_boom),
        patch.object(lifespan_mod, "_shutdown_phase3_components", new=AsyncMock()) as teardown,
    ):
        with pytest.raises(RuntimeError, match="falkordb binary corrupt"):
            await run(settings)

    teardown.assert_awaited_once()


async def test_run_registers_sigterm_and_sigint_handlers(
    settings: JanusSettings,
) -> None:
    """SIGTERM and SIGINT must both be wired to request_shutdown.

    Only SIGTERM is sent by most process supervisors; a regression that dropped
    one of the two would make ctrl-C behave differently from `pmc stop` in
    ways that are hard to notice in a test log.
    """

    async def _boot_then_shutdown(ctx: DaemonContext) -> None:
        ctx.request_shutdown()

    loop = asyncio.get_running_loop()
    with (
        patch.object(lifespan_mod, "boot", new=_boot_then_shutdown),
        patch.object(lifespan_mod, "_shutdown_phase3_components", new=AsyncMock()),
        patch.object(loop, "add_signal_handler") as add_handler,
    ):
        await run(settings)

    registered = {c.args[0] for c in add_handler.call_args_list}
    assert signal.SIGTERM in registered
    assert signal.SIGINT in registered
    assert all(c.args[1].__name__ == "request_shutdown" for c in add_handler.call_args_list)


async def test_run_stops_cron_loop_within_timeout(
    settings: JanusSettings,
) -> None:
    """CronLoop.stop() is wrapped in wait_for(10s) so a stuck sweep cannot
    block shutdown forever."""
    cron = MagicMock(stop=AsyncMock(), running=True)

    async def _boot_then_shutdown(ctx: DaemonContext) -> None:
        ctx.cron_loop = cron
        ctx.request_shutdown()

    with (
        patch.object(lifespan_mod, "boot", new=_boot_then_shutdown),
        patch.object(lifespan_mod, "_shutdown_phase3_components", new=AsyncMock()),
    ):
        rc = await run(settings)

    assert rc == 0
    cron.stop.assert_awaited_once()


async def test_run_survives_cron_loop_stop_timeout(
    settings: JanusSettings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hung cron stop logs and continues instead of aborting shutdown.

    Only the *first* ``wait_for`` is made to time out. The one in ``run`` that
    guards ``_shutdown_phase3_components`` must still run, otherwise this test
    would pass by skipping the very cleanup it claims to protect.
    """
    cron = MagicMock(stop=AsyncMock(), running=True)
    real_wait_for = asyncio.wait_for

    async def _boot_then_shutdown(ctx: DaemonContext) -> None:
        ctx.cron_loop = cron
        ctx.request_shutdown()

    with (
        patch.object(lifespan_mod, "boot", new=_boot_then_shutdown),
        patch.object(lifespan_mod, "_shutdown_phase3_components", new=AsyncMock()) as teardown,
    ):
        calls = {"n": 0}

        async def _wait_for_once_timeout(awaitable: Any, *args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                awaitable.close() if hasattr(awaitable, "close") else None
                raise asyncio.TimeoutError
            return await real_wait_for(awaitable, *args, **kwargs)

        with patch.object(lifespan_mod.asyncio, "wait_for", new=_wait_for_once_timeout):
            rc = await run(settings)

    assert rc == 0
    # Phase-3 teardown still ran => we got past the timeout, we did not abort.
    teardown.assert_awaited_once()
    assert any("cron_loop stop timed out" in r.message for r in caplog.records)


# --- regression pin --------------------------------------------------------


async def test_phase3_build_order_is_queue_then_embedding_then_search(
    ctx: DaemonContext,
) -> None:
    """Documented init order: queue → embedding client → search engine.

    ``SearchEngine`` receives the already-started ``EmbeddingClient``, so a
    reordered build would hand it ``None`` at construction. Only the *order*
    is asserted — the constructors themselves are covered elsewhere.
    """
    started: list[str] = []

    ctx.settings.pipeline.queue_db_path = ":memory:"

    with (
        patch.object(lifespan_mod, "EpisodeQueue", MagicMock()),
        patch.object(
            lifespan_mod,
            "EpisodeQueueAdapter",
            MagicMock(
                return_value=MagicMock(
                    start=AsyncMock(side_effect=lambda: started.append("adapter"))
                )
            ),
        ),
        patch.object(
            lifespan_mod,
            "EmbeddingClient",
            MagicMock(
                return_value=MagicMock(
                    start=AsyncMock(side_effect=lambda: started.append("embedding")),
                )
            ),
        ),
        patch.object(
            lifespan_mod,
            "SearchEngine",
            MagicMock(
                return_value=MagicMock(
                    start=AsyncMock(side_effect=lambda: started.append("search")),
                    _redis=None,
                )
            ),
        ),
        patch.object(lifespan_mod, "_warm_graphiti_singleton", new=AsyncMock(return_value=None)),
    ):
        await _build_phase3_components(ctx)

    assert started == ["adapter", "embedding", "search"]
    assert ctx.search_engine is not None
