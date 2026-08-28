"""Migration and snapshot utilities for transitioning from legacy graphiti configurations to Janus-Graph."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import platform
import shutil
import sqlite3
from typing import Any, Dict, Optional
import yaml

logger = logging.getLogger("janus_graph.migrate")


def detect_artifact(system: Optional[str] = None, machine: Optional[str] = None) -> Dict[str, Any]:
    """Auto-detect OS and architecture artifact requirements for Janus-Graph."""
    sys_name = (system or platform.system()).lower()
    mach_name = (machine or platform.machine()).lower()

    # Normalize architecture names
    if mach_name in ("aarch64", "arm64"):
        norm_arch = "arm64"
    elif mach_name in ("x86_64", "amd64"):
        norm_arch = "x86_64"
    else:
        norm_arch = mach_name

    artifact_id = f"falkordb-{sys_name}-{norm_arch}"
    supported = (sys_name in ("linux", "darwin")) and (norm_arch in ("arm64", "x86_64"))

    return {
        "system": sys_name,
        "machine": mach_name,
        "normalized_arch": norm_arch,
        "artifact": artifact_id,
        "supported": supported,
    }


def convert_legacy_dict(legacy_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert legacy graphiti_config.json dictionary to Janus-Graph config structure."""
    converted: Dict[str, Any] = {
        "engine": {},
        "graphiti": {
            "llm": {},
            "embedding": {},
        },
        "pipeline": {
            "queue_db_path": "./data/episodes.db",
            "worker_concurrency": 6,
            "max_attempts": 3,
            "cron_interval_min": 10,
        },
        "heuristics": {
            "auto_repair": True,
            "active_rules": ["edge_duplicate", "extracted_edges", "node_resolutions"],
        },
        "cache": {
            "embed": {
                "enabled": True,
                "max_size": 10000,
                "eviction": "lru",
            }
        },
        "report": {
            "min_severity": "info",
            "sinks": {
                "file": {"enabled": True, "path": "./data/logs/janus_report.jsonl"},
                "cli": {"enabled": True, "format": "pretty", "min_severity": "warning"},
            },
        },
    }

    # Database / Engine mapping
    db = legacy_data.get("database", {})
    if "host" in db:
        converted["engine"]["host"] = db["host"]
    if "port" in db:
        converted["engine"]["port"] = int(db["port"])
    if "database" in db:
        converted["graphiti"]["group_id"] = db["database"]

    # LLM mapping
    llm = legacy_data.get("llm", {})
    if "api_base" in llm:
        converted["graphiti"]["llm"]["base_url"] = llm["api_base"]
    elif "base_url" in llm:
        converted["graphiti"]["llm"]["base_url"] = llm["base_url"]
    if "model" in llm:
        converted["graphiti"]["llm"]["model"] = llm["model"]
    if "api_key" in llm:
        converted["graphiti"]["llm"]["api_key"] = llm["api_key"]
    if "temperature" in llm:
        converted["graphiti"]["llm"]["temperature"] = float(llm["temperature"])
    if "max_tokens" in llm:
        converted["graphiti"]["llm"]["max_tokens"] = int(llm["max_tokens"])

    # Embedding mapping
    emb = legacy_data.get("embedding", {})
    if "api_base" in emb:
        converted["graphiti"]["embedding"]["base_url"] = emb["api_base"]
    elif "base_url" in emb:
        converted["graphiti"]["embedding"]["base_url"] = emb["base_url"]
    if "model" in emb:
        converted["graphiti"]["embedding"]["model"] = emb["model"]
    if "api_key" in emb:
        converted["graphiti"]["embedding"]["api_key"] = emb["api_key"]
    if "embedding_dim" in emb:
        converted["graphiti"]["embedding"]["dim"] = int(emb["embedding_dim"])
    elif "dim" in emb:
        converted["graphiti"]["embedding"]["dim"] = int(emb["dim"])

    return converted


def json_to_yaml(json_path: Path | str, yaml_path: Path | str) -> Dict[str, Any]:
    """Migrate a legacy JSON config file into a Janus-Graph YAML configuration file."""
    j_path = Path(json_path)
    y_path = Path(yaml_path)

    if not j_path.exists():
        raise FileNotFoundError(f"Source JSON config not found: {j_path}")

    with open(j_path, "r", encoding="utf-8") as f:
        legacy_dict = json.load(f)

    converted = convert_legacy_dict(legacy_dict)

    y_path.parent.mkdir(parents=True, exist_ok=True)
    with open(y_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(converted, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Successfully migrated config from {j_path} to {y_path}")
    return converted


def snapshot_database(
    source_db_path: Path | str,
    target_dir: Path | str,
    include_checkpoints: bool = True,
) -> Dict[str, Any]:
    """Create an atomic snapshot of the SQLite database and checkpoints directory."""
    src = Path(source_db_path)
    dst_dir = Path(target_dir)

    if not src.exists():
        raise FileNotFoundError(f"Source database not found: {src}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_db = dst_dir / "episodes.db"

    # Use SQLite online backup API for transactionally safe, atomic snapshot
    src_conn = sqlite3.connect(str(src))
    dst_conn = sqlite3.connect(str(dst_db))
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()

    # Calculate SHA256 of backed up database
    with open(dst_db, "rb") as f:
        db_hash = hashlib.sha256(f.read()).hexdigest()

    # Calculate record counts
    chk_conn = sqlite3.connect(str(dst_db))
    counts: Dict[str, int] = {"total": 0, "queued": 0, "done": 0, "failed": 0, "dlq": 0}
    try:
        cur = chk_conn.cursor()
        cur.execute("SELECT COUNT(*) FROM episodes;")
        counts["total"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM episodes WHERE status='queued';")
        counts["queued"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM episodes WHERE status='done';")
        counts["done"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM episodes WHERE status='failed';")
        counts["failed"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM episodes WHERE status IN ('failed', 'aborted', 'dead_letter');")
        counts["dlq"] = cur.fetchone()[0]
    except Exception as e:
        logger.warning(f"Could not calculate counts from snapshot db: {e}")
    finally:
        chk_conn.close()

    # Copy checkpoints if requested and present
    src_chk = src.parent / "checkpoints"
    if include_checkpoints and src_chk.exists() and src_chk.is_dir():
        dst_chk = dst_dir / "checkpoints"
        if dst_chk.exists():
            shutil.rmtree(dst_chk)
        shutil.copytree(src_chk, dst_chk)

    metadata = {
        "source_db": str(src.resolve()),
        "target_db": str(dst_db.resolve()),
        "sha256": db_hash,
        "counts": counts,
    }

    with open(dst_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def rollback_database(
    snapshot_dir: Path | str,
    target_data_dir: Path | str,
) -> Dict[str, Any]:
    """Restore database and checkpoints from a snapshot directory to the target data directory."""
    s_dir = Path(snapshot_dir)
    t_dir = Path(target_data_dir)

    if not s_dir.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {s_dir}")

    s_db = s_dir / "episodes.db"
    if not s_db.exists():
        # check latest subdir
        s_db = s_dir / "latest" / "episodes.db"
        if not s_db.exists():
            raise FileNotFoundError(f"episodes.db not found in snapshot: {s_dir}")

    t_dir.mkdir(parents=True, exist_ok=True)
    t_db = t_dir / "episodes.db"
    shutil.copy2(s_db, t_db)

    # Clean up stale WAL / SHM files if any
    for ext in ("-wal", "-shm"):
        wal_file = Path(f"{t_db}{ext}")
        if wal_file.exists():
            wal_file.unlink()

    # Also keep queue.db symlink/copy for compatibility
    q_db = t_dir / "queue.db"
    shutil.copy2(t_db, q_db)

    # Restore checkpoints if available
    s_chk = s_dir / "checkpoints"
    if not s_chk.exists() and (s_dir / "latest" / "checkpoints").exists():
        s_chk = s_dir / "latest" / "checkpoints"

    if s_chk.exists() and s_chk.is_dir():
        t_chk = t_dir / "checkpoints"
        if t_chk.exists():
            shutil.rmtree(t_chk)
        shutil.copytree(s_chk, t_chk)

    return {
        "restored_db": str(t_db.resolve()),
        "status": "success",
    }
