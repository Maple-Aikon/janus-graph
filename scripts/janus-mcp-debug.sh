#!/bin/bash
# /home/maple/projects/janus-graph/scripts/janus-mcp-debug.sh
# Capture janus-graph MCP startup stderr for 10s
set +e
cd /home/maple/projects/janus-graph
exec /home/maple/.local/bin/uv run janus-graph --config /home/maple/projects/janus-graph/config.yaml mcp
