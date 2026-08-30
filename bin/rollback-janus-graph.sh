#!/usr/bin/env bash
# Rollback janus-graph state to snapshot and restore legacy routing if needed
set -euo pipefail

SNAPSHOT_DIR="${1:?Usage: $0 <snapshot_dir> [target_data_dir]}"
TARGET_DATA_DIR="${2:-./data}"

if [[ ! -d "$SNAPSHOT_DIR" ]]; then
    echo "Error: Snapshot directory not found: $SNAPSHOT_DIR" >&2
    exit 1
fi

SNAP_DB="$SNAPSHOT_DIR/episodes.db"
if [[ ! -f "$SNAP_DB" && -f "$SNAPSHOT_DIR/latest/episodes.db" ]]; then
    SNAP_DB="$SNAPSHOT_DIR/latest/episodes.db"
fi

if [[ ! -f "$SNAP_DB" ]]; then
    echo "Error: episodes.db not found in snapshot dir: $SNAPSHOT_DIR" >&2
    exit 1
fi

echo "==> Restoring episodes.db to $TARGET_DATA_DIR/episodes.db..."
mkdir -p "$TARGET_DATA_DIR"
cp "$SNAP_DB" "$TARGET_DATA_DIR/episodes.db"
cp "$SNAP_DB" "$TARGET_DATA_DIR/queue.db" 2>/dev/null || true

# Remove stale WAL / SHM files
rm -f "$TARGET_DATA_DIR/episodes.db-wal" "$TARGET_DATA_DIR/episodes.db-shm" \
      "$TARGET_DATA_DIR/queue.db-wal" "$TARGET_DATA_DIR/queue.db-shm"

if [[ -d "$SNAPSHOT_DIR/checkpoints" ]]; then
    echo "==> Restoring checkpoints..."
    mkdir -p "$TARGET_DATA_DIR/checkpoints"
    cp -r "$SNAPSHOT_DIR/checkpoints/"* "$TARGET_DATA_DIR/checkpoints/" 2>/dev/null || true
fi

echo "==> Rollback restored successfully."
