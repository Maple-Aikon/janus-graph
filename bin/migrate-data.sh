#!/usr/bin/env bash
# Safely migrate legacy queue and checkpoints into Janus-Graph data directory
set -euo pipefail

LEGACY_DIR="${1:-/home/maple/.picoclaw/workspace/apps/graphiti-mcp/queue}"
TARGET_DIR="${2:-/home/maple/.picoclaw/workspace/sources/janus-graph/data}"
BACKUP_BASE="${3:-/home/maple/.picoclaw/workspace/state/migration}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve sqlite3 binary safely
if [[ -x "/usr/bin/sqlite3" ]]; then
    SQLITE_BIN="/usr/bin/sqlite3"
elif command -v sqlite3 &>/dev/null; then
    SQLITE_BIN="$(command -v sqlite3)"
else
    echo "Error: sqlite3 binary not found" >&2
    exit 1
fi

echo "==> Step 1: Pre-migration snapshot..."
if [[ -f "$LEGACY_DIR/episodes.db" ]]; then
    bash "$SCRIPT_DIR/snapshot-episodes-db.sh" "$BACKUP_BASE" "$LEGACY_DIR/episodes.db"
fi

echo "==> Step 2: Copying data files to $TARGET_DIR..."
mkdir -p "$TARGET_DIR"

if [[ -f "$LEGACY_DIR/episodes.db" ]]; then
    "$SQLITE_BIN" "$LEGACY_DIR/episodes.db" ".backup '$TARGET_DIR/episodes.db'"
    # Provide backward/forward compatibility name queue.db
    cp "$TARGET_DIR/episodes.db" "$TARGET_DIR/queue.db"
fi

if [[ -d "$LEGACY_DIR/checkpoints" ]]; then
    mkdir -p "$TARGET_DIR/checkpoints"
    cp -r "$LEGACY_DIR/checkpoints/"* "$TARGET_DIR/checkpoints/" 2>/dev/null || true
fi

echo "==> Step 3: Verifying data migration..."
if [[ -f "$TARGET_DIR/episodes.db" ]]; then
    COUNT=$("$SQLITE_BIN" "$TARGET_DIR/episodes.db" "SELECT COUNT(*) FROM episodes;" 2>/dev/null || echo 0)
    echo "==> Migration complete: $COUNT episodes active in $TARGET_DIR/episodes.db"
fi
