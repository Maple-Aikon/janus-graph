"""janus-graph daemon entrypoint.

Usage:
    python -m janus_graph.daemon              # uses config.yaml
    JANUS_CONFIG_PATH=/path/to/cfg.yaml \\
        python -m janus_graph.daemon          # explicit config

Phase 2 PR scope (no cron — B3 hard rule):
  - HTTP server on 127.0.0.1:8765 with GET /health
  - Falkor supervisor with circuit-breaker
  - SIGTERM/SIGINT graceful shutdown
"""

from __future__ import annotations

import asyncio
import logging
import sys

from ..config import JanusSettings
from .lifespan import run


def _configure_logging() -> None:
    """Phase 2: simple stderr logging. Phase 4 may add file sink via cfg.report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main() -> int:
    """Parse config + enter asyncio loop.

    Logs the effective ``JANUS_GRAPH_HOME`` so operators can verify which
    directory the daemon is using at boot. See ``_resolve_home`` in
    ``config.py`` for the resolution precedence.
    """
    _configure_logging()
    settings = JanusSettings()
    from ..config import _resolve_home
    home = _resolve_home()
    import os as _os
    logging.getLogger("janus_graph.daemon").info(
        "JANUS_GRAPH_HOME=%s (resolved from %s)",
        home,
        "env" if _os.getenv("JANUS_GRAPH_HOME") else "default(~/.janus-graph)",
    )
    return asyncio.run(run(settings))


if __name__ == "__main__":
    sys.exit(main())
