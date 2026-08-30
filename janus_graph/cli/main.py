"""Main CLI entrypoint for janus-graph."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Optional

from ..config import JanusSettings, load_config
from ..engine.server import FalkorDBServerManager
from ..pipeline.cron import run_cron_sweep
from ..pipeline.dream import run_dream_consolidation
from ..pipeline.queue import EpisodeQueue
from ..report.dispatcher import ReportDispatcher
from ..report.models import ReportEvent, ReportSeverity
from ..migrate import json_to_yaml, snapshot_database, rollback_database, detect_artifact

logger = logging.getLogger("janus_graph.cli")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="janus-graph",
        description="Janus-Graph Knowledge Graph Memory Engine CLI",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None, help="Path to configuration file (YAML or JSON)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # doctor
    subparsers.add_parser("doctor", help="Run environment and service diagnostics")

    # engine
    engine_parser = subparsers.add_parser("engine", help="Manage FalkorDB backend instance")
    engine_parser.add_argument(
        "action", choices=["start", "stop", "status", "restart"], help="Engine action"
    )

    # sweep
    sweep_parser = subparsers.add_parser("sweep", help="Process queued episodes via cron sweep")
    sweep_parser.add_argument("--batch-size", type=int, default=None, help="Max episodes to process")

    # dream
    dream_parser = subparsers.add_parser("dream", help="Execute dream mode memory consolidation")
    dream_parser.add_argument("--force", action="store_true", help="Force community clustering")
    dream_parser.add_argument("--group-id", type=str, default=None, help="Target memory group")

    # queue
    queue_parser = subparsers.add_parser("queue", help="Inspect and manage episode queue and DLQ")
    queue_subs = queue_parser.add_subparsers(dest="queue_action", help="Queue action")
    queue_subs.add_parser("status", help="Show queue statistics and DLQ count")
    retry_parser = queue_subs.add_parser("retry", help="Replay a dead-letter episode")
    retry_parser.add_argument("episode_id", help="Episode ID to replay")
    queue_subs.add_parser("reap", help="Reap stuck or failed/aborted episodes")

    # cache
    cache_parser = subparsers.add_parser("cache", help="Inspect and manage caches")
    cache_subs = cache_parser.add_subparsers(dest="cache_action", help="Cache action")
    cache_subs.add_parser("stats", help="Show embedding cache stats")

    # report
    report_parser = subparsers.add_parser("report", help="Inspect execution reports")
    report_subs = report_parser.add_subparsers(dest="report_action", help="Report action")
    report_stats_parser = report_subs.add_parser("stats", help="Show recent report log summaries")
    report_stats_parser.add_argument("--limit", type=int, default=10, help="Number of reports to show")
    report_stats_parser.add_argument("--kind", type=str, default=None, help="Filter by report kind")
    report_subs.add_parser("test-telegram", help="Send a test alert through Telegram sink")
    report_subs.add_parser("test-webhook", help="Send a test event through Webhook sink")

    # migrate
    migrate_parser = subparsers.add_parser("migrate", help="Convert legacy JSON config to YAML")
    migrate_parser.add_argument(
        "--json-path", default="graphiti_config.json", help="Source JSON config file"
    )
    migrate_parser.add_argument(
        "--yaml-path", default="config.yaml", help="Target YAML config file"
    )

    # snapshot
    snapshot_parser = subparsers.add_parser("snapshot", help="Create an atomic backup of queue database and checkpoints")
    snapshot_parser.add_argument(
        "--target-dir", default="./data/snapshots", help="Target snapshot directory"
    )
    snapshot_parser.add_argument(
        "--source-db", default=None, help="Source SQLite DB path"
    )

    # rollback
    rollback_parser = subparsers.add_parser("rollback", help="Restore database and checkpoints from a snapshot")
    rollback_parser.add_argument(
        "snapshot_dir", help="Path to snapshot directory containing episodes.db"
    )
    rollback_parser.add_argument(
        "--target-data-dir", default=None, help="Target data directory"
    )

    # mcp
    subparsers.add_parser("mcp", help="Start stdio MCP server for agent integration")

    return parser


def run_cli(args: argparse.Namespace, cfg: JanusSettings) -> int:
    """Execute parsed CLI command."""
    cmd = args.command
    if not cmd:
        return 0

    if cmd == "doctor":
        print("🏛️ Janus-Graph Diagnostics")
        print(f"  • Engine Host/Port: {cfg.engine.host}:{cfg.engine.port}")
        print(f"  • Queue DB: {cfg.pipeline.queue_db_path}")
        
        # Check SQLite
        try:
            queue = EpisodeQueue(cfg.pipeline.queue_db_path)
            stats = queue.get_stats()
            print(f"  • SQLite Queue Status: OK (Stats: {stats})")
        except Exception as e:
            print(f"  • SQLite Queue Status: ERROR ({e})")

        # Check FalkorDB engine reachability
        try:
            from falkordb import FalkorDB
            fdb = FalkorDB(host=cfg.engine.host, port=cfg.engine.port)
            fdb.connection.ping()
            print("  • FalkorDB Engine: ONLINE")
        except Exception as e:
            print(f"  • FalkorDB Engine: OFFLINE ({e})")

        # Check Host Architecture
        arch_info = detect_artifact()
        print(f"  • Host System: {arch_info['system']} ({arch_info['normalized_arch']}) [Artifact: {arch_info['artifact']}, Supported: {arch_info['supported']}]")

        # Check LLM / Embedding Config
        print(f"  • LLM Model: {cfg.graphiti.llm.model} ({cfg.graphiti.llm.base_url})")
        print(f"  • Embedding Model: {cfg.graphiti.embedding.model} (dim={cfg.graphiti.embedding.dim})")
        print(f"  • Active Heuristic Rules: {cfg.heuristics.active_rules}")
        print("✅ Doctor diagnostics finished.")
        return 0

    elif cmd == "engine":
        mgr = FalkorDBServerManager(cfg.engine)
        if args.action == "start":
            res = mgr.start()
            print(f"Engine start result: {res}")
            return 0 if res else 1
        elif args.action == "stop":
            res = mgr.stop()
            print(f"Engine stop result: {res}")
            return 0 if res else 1
        elif args.action == "status":
            running = mgr.is_running()
            print(f"Engine running: {running}")
            return 0
        elif args.action == "restart":
            mgr.stop()
            res = mgr.start()
            print(f"Engine restart result: {res}")
            return 0 if res else 1

    elif cmd == "sweep":
        print("🔄 Starting cron sweep...")
        res = asyncio.run(run_cron_sweep(cfg, batch_size=args.batch_size))
        summary_text = f"Sweep completed: {res.get('succeeded', 0)}/{res.get('processed', 0)} ok, {res.get('failed', 0)} fail, {res.get('queued_remaining', 0)} left ({res.get('duration_ms', 0)}ms)"
        print(summary_text)
        return 0

    elif cmd == "dream":
        print(f"🌙 Starting dream consolidation (force={args.force})...")
        res = asyncio.run(run_dream_consolidation(cfg, force=args.force, group_id=args.group_id))
        summary_text = f"Dream consolidation completed: {res.get('status', 'ok')} ({res.get('duration_ms', 0)}ms)"
        print(summary_text)
        return 0

    elif cmd == "queue":
        queue = EpisodeQueue(cfg.pipeline.queue_db_path)
        action = getattr(args, "queue_action", None) or "status"
        if action == "status":
            stats = queue.get_stats()
            print("📦 Episode Queue Status:")
            for k, v in stats.items():
                print(f"  • {k}: {v}")
            dlq = queue.get_dlq_records(limit=5)
            if dlq:
                print(f"⚠️ Recent DLQ Entries ({len(dlq)}):")
                for r in dlq:
                    print(f"  - [{r.get('episode_id')}] attempts={r.get('attempt_count')} error={r.get('last_error')}")
            return 0
        elif action == "retry":
            ok = asyncio.run(queue.replay_dlq_episode(args.episode_id))
            if ok:
                print(f"✅ Episode {args.episode_id} re-queued successfully.")
                return 0
            else:
                print(f"❌ Episode {args.episode_id} not found in DLQ.")
                return 1
        elif action == "reap":
            stuck = asyncio.run(queue.reap_stuck_processing(timeout_sec=cfg.pipeline.attempt_timeout_sec, max_attempts=cfg.pipeline.max_attempts))
            reaped = asyncio.run(queue.reap_failed_or_aborted())
            print(f"🧹 Queue reaped: {stuck} stuck records recovered, {reaped} failed/aborted records re-queued.")
            return 0

    elif cmd == "cache":
        action = getattr(args, "cache_action", None) or "stats"
        if action == "stats":
            print("⚡ Embedding Cache Configuration:")
            print(f"  • Enabled: {cfg.cache.embed.enabled}")
            print(f"  • Max Size: {cfg.cache.embed.max_size}")
            print(f"  • Eviction Policy: {cfg.cache.embed.eviction}")
            return 0

    elif cmd == "report":
        action = getattr(args, "report_action", None) or "stats"
        if action == "stats":
            path = Path(cfg.report.sinks.file.path)
            if not path.exists():
                print(f"No reports found at {path}")
                return 0
            print(f"📊 Recent Reports from {path}:\n")
            lines = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if args.kind and item.get("kind") != args.kind:
                            continue
                        lines.append(item)
                    except json.JSONDecodeError:
                        continue
            for item in lines[-args.limit:]:
                ts = item.get("timestamp", "")
                kind = item.get("kind", "")
                sev = item.get("severity", "").upper()
                summary = item.get("summary", "")
                print(f"[{ts}] [{sev}] {kind}: {summary}")
            return 0
        elif action == "test-telegram":
            dispatcher = ReportDispatcher.from_settings(cfg)
            report = ReportEvent(
                kind="test_alert",
                severity=ReportSeverity.INFO,
                summary="Janus-Graph CLI Telegram test message",
                details={"sender": "cli", "status": "ok"},
            )
            asyncio.run(dispatcher.dispatch(report))
            print("✅ Dispatched test alert to Telegram sink.")
            return 0
        elif action == "test-webhook":
            dispatcher = ReportDispatcher.from_settings(cfg)
            report = ReportEvent(
                kind="test_alert",
                severity=ReportSeverity.INFO,
                summary="Janus-Graph CLI Webhook test event",
                details={"sender": "cli", "event": "ping"},
            )
            asyncio.run(dispatcher.dispatch(report))
            print("✅ Dispatched test event to Webhook sink.")
            return 0

    elif cmd == "migrate":
        print(f"🚚 Migrating config: {args.json_path} -> {args.yaml_path}")
        try:
            json_to_yaml(args.json_path, args.yaml_path)
            print(f"✅ Successfully created {args.yaml_path}")
            return 0
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return 1

    elif cmd == "snapshot":
        src_db = args.source_db or cfg.pipeline.queue_db_path
        print(f"📸 Creating snapshot of {src_db} into {args.target_dir}...")
        try:
            meta = snapshot_database(src_db, args.target_dir)
            print(f"✅ Snapshot created successfully: {meta['counts']['total']} total records, sha256={meta['sha256'][:12]}...")
            return 0
        except Exception as e:
            print(f"❌ Snapshot failed: {e}")
            return 1

    elif cmd == "rollback":
        target_dir = args.target_data_dir or str(Path(cfg.pipeline.queue_db_path).parent)
        print(f"⏪ Restoring snapshot from {args.snapshot_dir} into {target_dir}...")
        try:
            res = rollback_database(args.snapshot_dir, target_dir)
            print(f"✅ Rollback restored successfully: {res['restored_db']}")
            return 0
        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            return 1

    elif cmd == "mcp":
        from ..mcp.server import create_mcp_server
        server = create_mcp_server(cfg)
        server.run(transport="stdio")
        return 0

    return 0


def cli_entrypoint() -> None:
    """Console script entry point."""
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    cfg = load_config(args.config)
    exit_code = run_cli(args, cfg)
    sys.exit(exit_code)


if __name__ == "__main__":
    cli_entrypoint()
