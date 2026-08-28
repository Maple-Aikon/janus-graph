"""In-memory LRU Cache Backend."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional
from .base import CacheBackend


class LRUCacheBackend(CacheBackend):
    """OrderedDict-backed thread-safe LRU cache."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)
