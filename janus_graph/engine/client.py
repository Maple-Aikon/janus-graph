"""FalkorDB client connection factory and health checks."""

from __future__ import annotations

from typing import Any, Optional
from ..config import EngineConfig


class FalkorDBClientFactory:
    """Factory for FalkorDB connections."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()

    def get_client(self) -> Any:
        """Create and return a FalkorDB client instance."""
        from falkordb import FalkorDB
        return FalkorDB(host=self.config.host, port=self.config.port)

    def ping(self) -> bool:
        """Check if FalkorDB responds to ping."""
        try:
            client = self.get_client()
            return bool(client.connection.ping())
        except Exception:
            return False
