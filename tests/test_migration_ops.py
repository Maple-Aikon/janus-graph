"""Phase 5: Snapshot, rollback, and data migration ops tests."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from janus_graph.migrate import snapshot_database, rollback_database


def _create_test_db(path: Path, num_records: int = 10) -> None:
    """Build a small SQLite database matching EpisodeQueue schema."""
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT
        )
        """
    )
    for i in range(num_records):
        cur.execute(
            "INSERT INTO episodes (episode_id, content, status) VALUES (?, ?, ?)",
            (f"ep-{i:04d}", f"test content {i}", "queued" if i % 2 == 0 else "done"),
        )
    conn.commit()
    conn.close()


def test_snapshot_atomic_creates_episodes_db(temp_dir: Path) -> None:
    """snapshot_database creates episodes.db + metadata.json + checkpoints directory."""
    src_db = temp_dir / "episodes.db"
    _create_test_db(src_db, num_records=5)

    snap_dir = temp_dir / "snap"
    meta = snapshot_database(src_db, snap_dir)

    assert (snap_dir / "episodes.db").exists()
    assert (snap_dir / "metadata.json").exists()
    assert meta["counts"]["total"] == 5
    assert meta["counts"]["queued"] == 3
    assert meta["counts"]["done"] == 2
    assert len(meta["sha256"]) == 64


def test_snapshot_includes_checkpoints(temp_dir: Path) -> None:
    """When checkpoints/ sibling dir exists, snapshot includes it."""
    src_db = temp_dir / "episodes.db"
    _create_test_db(src_db, num_records=2)
    chk_src = temp_dir / "checkpoints"
    chk_src.mkdir()
    (chk_src / "ckpt1.json").write_text("{}")

    snap_dir = temp_dir / "snap"
    snapshot_database(src_db, snap_dir, include_checkpoints=True)

    assert (snap_dir / "checkpoints" / "ckpt1.json").exists()


def test_snapshot_missing_source_raises(temp_dir: Path) -> None:
    """Source DB missing → FileNotFoundError."""
    src_db = temp_dir / "missing.db"
    snap_dir = temp_dir / "snap"
    with pytest.raises(FileNotFoundError):
        snapshot_database(src_db, snap_dir)


def test_rollback_restores_episodes_db(temp_dir: Path) -> None:
    """rollback_database copies snapshot episodes.db into target_data_dir."""
    src_db = temp_dir / "src_episodes.db"
    _create_test_db(src_db, num_records=3)

    snap_dir = temp_dir / "snap"
    snapshot_database(src_db, snap_dir)

    target_data = temp_dir / "data"
    result = rollback_database(snap_dir, target_data)

    restored_db = Path(result["restored_db"])
    assert restored_db.exists()
    conn = sqlite3.connect(str(restored_db))
    count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    conn.close()
    assert count == 3


def test_rollback_resolves_latest_subdir(temp_dir: Path) -> None:
    """Snapshot dir with `latest/` subdir is auto-resolved."""
    src_db = temp_dir / "src_episodes.db"
    _create_test_db(src_db, num_records=1)

    snap_latest = temp_dir / "snap" / "latest"
    snapshot_database(src_db, snap_latest)

    target_data = temp_dir / "data"
    result = rollback_database(temp_dir / "snap", target_data)
    assert Path(result["restored_db"]).exists()


def test_rollback_missing_snapshot_raises(temp_dir: Path) -> None:
    """Snapshot dir missing → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        rollback_database(temp_dir / "ghost", temp_dir / "data")


def test_rollback_clears_stale_wal_shm(temp_dir: Path) -> None:
    """Stale WAL/SHM files in target_data are removed during rollback."""
    src_db = temp_dir / "src_episodes.db"
    _create_test_db(src_db, num_records=1)
    snap_dir = temp_dir / "snap"
    snapshot_database(src_db, snap_dir)

    target_data = temp_dir / "data"
    target_data.mkdir()
    (target_data / "episodes.db-wal").write_text("stale")
    (target_data / "episodes.db-shm").write_text("stale")

    rollback_database(snap_dir, target_data)

    assert not (target_data / "episodes.db-wal").exists()
    assert not (target_data / "episodes.db-shm").exists()


def test_snapshot_then_rollback_round_trip(temp_dir: Path) -> None:
    """End-to-end snapshot → modify → rollback returns to original state."""
    src_db = temp_dir / "episodes.db"
    _create_test_db(src_db, num_records=4)

    snap_dir = temp_dir / "snap"
    snapshot_database(src_db, snap_dir)

    # Modify original DB after snapshot
    conn = sqlite3.connect(str(src_db))
    conn.execute("UPDATE episodes SET status='done' WHERE episode_id='ep-0000'")
    conn.commit()
    conn.close()

    # Rollback should restore the snapshot state (still original)
    target_data = temp_dir / "data"
    rollback_database(snap_dir, target_data)

    conn = sqlite3.connect(str(target_data / "episodes.db"))
    status = conn.execute("SELECT status FROM episodes WHERE episode_id='ep-0000'").fetchone()[0]
    conn.close()
    # Snapshot state: ep-0000 was queued (0 % 2 == 0)
    assert status == "queued"


def test_snapshot_writes_valid_metadata_json(temp_dir: Path) -> None:
    """metadata.json contains required keys with correct types."""
    src_db = temp_dir / "episodes.db"
    _create_test_db(src_db, num_records=2)
    snap_dir = temp_dir / "snap"
    snapshot_database(src_db, snap_dir)

    meta_file = snap_dir / "metadata.json"
    assert meta_file.exists()
    data = json.loads(meta_file.read_text())
    assert "source_db" in data
    assert "sha256" in data
    assert "counts" in data
    assert isinstance(data["counts"], dict)
