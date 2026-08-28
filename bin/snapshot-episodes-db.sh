#!/usr/bin/env bash
# Snapshot episodes.db and checkpoints with atomicity and metadata
set -euo pipefail

TARGET_BASE="${1:?Usage: $0 <target_dir> [source_db_path]}"
SOURCE_DB="${2:-/home/maple/.picoclaw/workspace/apps/graphiti-mcp/queue/episodes.db}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TARGET_DIR="${TARGET_BASE}/snapshot_${TIMESTAMP}"
LATEST_DIR="${TARGET_BASE}/latest"

# Resolve sqlite3 binary safely
if [[ -x "/usr/bin/sqlite3" ]]; then
    SQLITE_BIN="/usr/bin/sqlite3"
elif command -v sqlite3 &>/dev/null; then
    SQLITE_BIN="$(command -v sqlite3)"
else
    echo "Error: sqlite3 binary not found" >&2
    exit 1
fi

mkdir -p "$TARGET_DIR"

if [[ ! -f "$SOURCE_DB" ]]; then
    echo "Error: Source database not found: $SOURCE_DB" >&2
    exit 1
fi

echo "==> Backing up database from $SOURCE_DB to $TARGET_DIR/episodes.db..."
"$SQLITE_BIN" "$SOURCE_DB" ".backup '$TARGET_DIR/episodes.db'"

SOURCE_DIR="$(dirname "$SOURCE_DB")"
if [[ -d "$SOURCE_DIR/checkpoints" ]]; then
    echo "==> Backing up checkpoints..."
    cp -r "$SOURCE_DIR/checkpoints" "$TARGET_DIR/"
fi

# Metadata generation
TOTAL_COUNT=$("$SQLITE_BIN" "$TARGET_DIR/episodes.db" "SELECT COUNT(*) FROM episodes;" 2>/dev/null || echo 0)
QUEUED_COUNT=$("$SQLITE_BIN" "$TARGET_DIR/episodes.db" "SELECT COUNT(*) FROM episodes WHERE status='queued';" 2>/dev/null || echo 0)
DONE_COUNT=$("$SQLITE_BIN" "$TARGET_DIR/episodes.db" "SELECT COUNT(*) FROM episodes WHERE status='done';" 2>/dev/null || echo 0)
DLQ_COUNT=$("$SQLITE_BIN" "$TARGET_DIR/episodes.db" "SELECT COUNT(*) FROM episodes WHERE status IN ('failed', 'aborted', 'dead_letter');" 2>/dev/null || echo 0)
DB_SHA256=$(sha256sum "$TARGET_DIR/episodes.db" | awk '{print $1}')

cat > "$TARGET_DIR/metadata.json" <<EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "source_db": "$SOURCE_DB",
  "sha256": "$DB_SHA256",
  "counts": {
    "total": $TOTAL_COUNT,
    "queued": $QUEUED_COUNT,
    "done": $DONE_COUNT,
    "dlq": $DLQ_COUNT
  }
}
EOF

# Update latest symlink / copy
mkdir -p "$LATEST_DIR"
cp "$TARGET_DIR/episodes.db" "$LATEST_DIR/episodes.db"
cp "$TARGET_DIR/metadata.json" "$LATEST_DIR/metadata.json"
if [[ -d "$TARGET_DIR/checkpoints" ]]; then
    cp -r "$TARGET_DIR/checkpoints" "$LATEST_DIR/"
fi

echo "==> Snapshot completed successfully at $TARGET_DIR (latest updated)"
