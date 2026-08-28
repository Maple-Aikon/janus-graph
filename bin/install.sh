#!/usr/bin/env bash
# Janus-Graph Engine Binary Installer & Verification
set -euo pipefail

BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="$(uname -m)"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

echo "==> Detecting host environment: ${OS} (${ARCH})"

copy_from_local_if_available() {
    local legacy_bin="/home/maple/.picoclaw/workspace/apps/falkordb-server/bin"
    if [[ -d "$legacy_bin" ]]; then
        if [[ ! -f "$BIN_DIR/falkordb.so" && -f "$legacy_bin/falkordb.so" ]]; then
            echo "==> Copying pre-built falkordb.so from local cache..."
            cp "$legacy_bin/falkordb.so" "$BIN_DIR/falkordb.so"
        fi
        if [[ ! -f "$BIN_DIR/redis-server-8" && -f "$legacy_bin/redis-server-8" ]]; then
            echo "==> Copying pre-built redis-server-8 from local cache..."
            cp "$legacy_bin/redis-server-8" "$BIN_DIR/redis-server-8"
        fi
    fi
}

copy_from_local_if_available

if [[ ! -f "$BIN_DIR/falkordb.so" ]]; then
    echo "==> Warning: bin/falkordb.so not found. Run ./bin/build.sh or download release artifact for ${ARCH}."
else
    chmod +x "$BIN_DIR/falkordb.so" 2>/dev/null || true
    echo "==> bin/falkordb.so is ready."
fi

if [[ ! -f "$BIN_DIR/redis-server-8" ]]; then
    echo "==> Warning: bin/redis-server-8 not found. Please install Redis/Valkey 8+ or download binary."
else
    chmod +x "$BIN_DIR/redis-server-8"
    echo "==> bin/redis-server-8 is ready."
fi

echo "==> Verification completed."
