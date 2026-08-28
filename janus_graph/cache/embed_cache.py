"""Embedder wrapper with LRU caching."""

from __future__ import annotations

import hashlib
from typing import Any, List, Optional
from .lru import LRUCacheBackend
from .stats import CacheStats


class EmbedCache:
    """Wraps Graphiti embedder to cache text vector embeddings."""

    def __init__(self, inner: Any, max_size: int = 10000):
        self.inner = inner
        self.backend = LRUCacheBackend(max_size=max_size)
        self.stats = CacheStats()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def embed(self, text: str) -> List[float]:
        key = self._hash_text(text)
        cached = self.backend.get(key)
        if cached is not None:
            self.stats.hits += 1
            return cached

        self.stats.misses += 1
        embedding = await self.inner.embed(text)
        self.backend.set(key, embedding)
        return embedding
