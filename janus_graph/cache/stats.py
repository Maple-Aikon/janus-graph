"""Cache metrics and statistics aggregator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
