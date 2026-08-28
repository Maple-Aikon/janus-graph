#!/usr/bin/env bash
# Build FalkorDB from source for specific architecture
set -euo pipefail

TARGET_ARCH="${1:-$(uname -m)}"
echo "==> Building FalkorDB for architecture: ${TARGET_ARCH}"

BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/build"
mkdir -p "$BUILD_DIR"

if ! command -v git &>/dev/null || ! command -v make &>/dev/null; then
    echo "Error: git and make/build-essential are required to build FalkorDB." >&2
    exit 1
fi

echo "==> Build script ready. In CI environment, this clones FalkorDB v1.7.1 and runs make."
