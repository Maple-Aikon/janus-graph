"""Main CLI entrypoint for janus-graph."""

from __future__ import annotations

import argparse
import asyncio
import sys
from ..config import load_config
from ..engine.server import FalkorDBServerManager
from ..pipeline.cron import run_cron_sweep
from ..pipeline.dream import run_dream_consolidation
from ..pipeline.queue import EpisodeQueue


def cli_entrypoint() -> None:
    """CLI dispatcher."""
    parser = argparse.ArgumentParser(
        prog="janus-graph",
        description="Janus-Graph Knowledge Graph Memory Engine CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # doctor
    subparsers.add_parser("doctor", help="Run system diagnostics and environment check")

    # engine
    engine_parser = subparsers.add_parser("engine", help="Manage FalkorDB engine")
    engine_parser.add_argument("action", choices=["start", "stop", "status"], help="Engine action")

    # sweep
    subparsers.add_parser("sweep", help="Process queued episodes")

    # dream
    dream_parser = subparsers.add_parser("dream", help="Run dream memory consolidation")
    dream_parser.add_argument("--force", action="store_true", help="Force clustering phase")

    # mcp
    subparsers.add_parser("mcp", help="Start FastMCP server")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    cfg = load_config()

    if args.command == "doctor":
        print("🏛️ Janus-Graph Diagnostics")
        print(f"Engine Port: {cfg.engine.port}")
        print(f"Queue DB: {cfg.pipeline.queue_db_path}")
        queue = EpisodeQueue(cfg.pipeline.queue_db_path)
        print(f"Queue Stats: {queue.get_stats()}")
        print("✅ Doctor diagnostics completed.")

    elif args.command == "engine":
        mgr = FalkorDBServerManager(cfg.engine)
        if args.action == "start":
            res = mgr.start()
            print(f"Engine started: {res}")
        elif args.action == "stop":
            res = mgr.stop()
            print(f"Engine stopped: {res}")
        elif args.action == "status":
            running = mgr.is_running()
            print(f"Engine running: {running}")

    elif args.command == "sweep":
        res = asyncio.run(run_cron_sweep(cfg))
        print(f"Sweep results: {res}")

    elif args.command == "dream":
        res = asyncio.run(run_dream_consolidation(cfg, force=args.force))
        print(f"Dream results: {res}")

    elif args.command == "mcp":
        from ..mcp.server import create_mcp_server
        server = create_mcp_server(cfg)
        server.run(transport="stdio")


if __name__ == "__main__":
    cli_entrypoint()
