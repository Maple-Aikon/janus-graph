"""Unit tests for Janus-Graph CLI and migration utilities."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from janus_graph.cli.main import build_parser, run_cli
from janus_graph.config import JanusSettings
from janus_graph.migrate import convert_legacy_dict, json_to_yaml, snapshot_database, rollback_database
from janus_graph.pipeline.queue import EpisodeQueue


def test_build_parser():
    parser = build_parser()

    args = parser.parse_args(["doctor"])
    assert args.command == "doctor"

    args = parser.parse_args(["engine", "status"])
    assert args.command == "engine"
    assert args.action == "status"

    args = parser.parse_args(["sweep", "--batch-size", "25"])
    assert args.command == "sweep"
    assert args.batch_size == 25

    args = parser.parse_args(["dream", "--force", "--group-id", "custom_group"])
    assert args.command == "dream"
    assert args.force is True
    assert args.group_id == "custom_group"

    args = parser.parse_args(["queue", "retry", "ep-123"])
    assert args.command == "queue"
    assert args.queue_action == "retry"
    assert args.episode_id == "ep-123"

    args = parser.parse_args(["report", "stats", "--limit", "5", "--kind", "cron_sweep"])
    assert args.command == "report"
    assert args.report_action == "stats"
    assert args.limit == 5
    assert args.kind == "cron_sweep"

    args = parser.parse_args(["migrate", "--json-path", "old.json", "--yaml-path", "new.yaml"])
    assert args.command == "migrate"
    assert args.json_path == "old.json"
    assert args.yaml_path == "new.yaml"

    args = parser.parse_args(["snapshot", "--target-dir", "./data/snap"])
    assert args.command == "snapshot"
    assert args.target_dir == "./data/snap"

    args = parser.parse_args(["rollback", "./data/snap/latest"])
    assert args.command == "rollback"
    assert args.snapshot_dir == "./data/snap/latest"


def test_migrate_convert_legacy_dict():
    legacy = {
        "database": {"host": "10.0.0.1", "port": 6380, "database": "test_db"},
        "llm": {"api_base": "http://llm:4000/v1", "model": "gpt-4o", "api_key": "sk-test", "temperature": 0.2, "max_tokens": 2048},
        "embedding": {"api_base": "http://emb:8081/v1", "model": "nomic-v2", "api_key": "sk-emb", "embedding_dim": 768},
    }
    converted = convert_legacy_dict(legacy)
    assert converted["engine"]["host"] == "10.0.0.1"
    assert converted["engine"]["port"] == 6380
    assert converted["graphiti"]["group_id"] == "test_db"
    assert converted["graphiti"]["llm"]["base_url"] == "http://llm:4000/v1"
    assert converted["graphiti"]["llm"]["model"] == "gpt-4o"
    assert converted["graphiti"]["embedding"]["dim"] == 768


def test_migrate_json_to_yaml(temp_dir: Path):
    j_file = temp_dir / "legacy.json"
    y_file = temp_dir / "out.yaml"

    j_file.write_text(json.dumps({
        "database": {"host": "localhost", "port": 6379, "database": "janus_test"},
        "llm": {"api_base": "http://localhost:4000", "model": "test-model"},
    }))

    res = json_to_yaml(j_file, y_file)
    assert y_file.exists()
    assert res["graphiti"]["group_id"] == "janus_test"


def test_cli_doctor(temp_dir: Path, capsys):
    db_path = temp_dir / "episodes.db"
    cfg = JanusSettings()
    cfg.pipeline.queue_db_path = str(db_path)

    parser = build_parser()
    args = parser.parse_args(["doctor"])

    code = run_cli(args, cfg)
    assert code == 0
    captured = capsys.readouterr().out
    assert "Janus-Graph Diagnostics" in captured
    assert "SQLite Queue Status: OK" in captured


def test_cli_queue_commands(temp_dir: Path, capsys):
    import asyncio
    db_path = temp_dir / "episodes.db"
    cfg = JanusSettings()
    cfg.pipeline.queue_db_path = str(db_path)

    queue = EpisodeQueue(str(db_path))
    ep_id = asyncio.run(queue.enqueue("Test CLI episode content", group_id="test_group"))

    parser = build_parser()
    args = parser.parse_args(["queue", "status"])
    code = run_cli(args, cfg)
    assert code == 0
    captured = capsys.readouterr().out
    assert "Episode Queue Status:" in captured
    assert "queued: 1" in captured

    args = parser.parse_args(["queue", "reap"])
    code = run_cli(args, cfg)
    assert code == 0
    assert "Queue reaped:" in capsys.readouterr().out

    args = parser.parse_args(["queue", "retry", ep_id])
    code = run_cli(args, cfg)
    assert code == 1


def test_cli_report_commands(temp_dir: Path, capsys):
    log_path = temp_dir / "reports.jsonl"
    cfg = JanusSettings()
    cfg.report.sinks.file.path = str(log_path)

    entry = {
        "timestamp": "2026-08-28T20:00:00Z",
        "kind": "cron_sweep",
        "severity": "info",
        "summary": "Processed 10 episodes",
    }
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    parser = build_parser()
    args = parser.parse_args(["report", "stats", "--limit", "5"])
    code = run_cli(args, cfg)
    assert code == 0
    captured = capsys.readouterr().out
    assert "Processed 10 episodes" in captured


def test_cli_cache_stats(capsys):
    cfg = JanusSettings()
    parser = build_parser()
    args = parser.parse_args(["cache", "stats"])
    code = run_cli(args, cfg)
    assert code == 0
    captured = capsys.readouterr().out
    assert "Embedding Cache Configuration" in captured


def test_cli_snapshot_command(temp_dir: Path, capsys):
    db_path = temp_dir / "episodes.db"
    snap_dir = temp_dir / "snap"
    cfg = JanusSettings()
    cfg.pipeline.queue_db_path = str(db_path)

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS episodes (episode_id TEXT PRIMARY KEY, content TEXT, status TEXT, attempt_count INTEGER DEFAULT 0, last_error TEXT)"
    )
    conn.execute("INSERT INTO episodes VALUES ('ep1', 'hi', 'queued', 0, NULL)")
    conn.commit()
    conn.close()

    parser = build_parser()
    args = parser.parse_args(["snapshot", "--target-dir", str(snap_dir)])
    code = run_cli(args, cfg)
    assert code == 0
    assert (snap_dir / "episodes.db").exists()
    captured = capsys.readouterr().out
    assert "Snapshot created" in captured


def test_cli_rollback_command(temp_dir: Path, capsys):
    db_path = temp_dir / "src.db"
    snap_dir = temp_dir / "snap"
    target_data = temp_dir / "data"
    cfg = JanusSettings()

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS episodes (episode_id TEXT PRIMARY KEY, content TEXT, status TEXT, attempt_count INTEGER DEFAULT 0, last_error TEXT)"
    )
    conn.execute("INSERT INTO episodes VALUES ('ep1', 'hi', 'queued', 0, NULL)")
    conn.commit()
    conn.close()

    snapshot_database(db_path, snap_dir)

    parser = build_parser()
    args = parser.parse_args(["rollback", str(snap_dir), "--target-data-dir", str(target_data)])
    code = run_cli(args, cfg)
    assert code == 0
    assert (target_data / "episodes.db").exists()
