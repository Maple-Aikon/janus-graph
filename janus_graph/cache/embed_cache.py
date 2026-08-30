"""Embedder wrapper with LRU caching."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, List, Optional
from graphiti_core.embedder.client import EmbedderClient
from .lru import LRUCacheBackend
from .stats import CacheStats


class EmbedCache(EmbedderClient):
    """Wraps Graphiti embedder to cache text vector embeddings."""

    def __init__(self, inner: Any, max_size: int = 10000):
        self.inner = inner
        self.backend = LRUCacheBackend(max_size=max_size)
        self.stats = CacheStats()

    @property
    def config(self) -> Any:
        return getattr(self.inner, "config", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_single_text(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> Optional[str]:
        if isinstance(input_data, str):
            return input_data
        if isinstance(input_data, list) and len(input_data) == 1 and isinstance(input_data[0], str):
            return input_data[0]
        return None

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        text = self._get_single_text(input_data)
        if text is not None:
            key = self._hash_text(text)
            cached = self.backend.get(key)
            if cached is not None:
                self.stats.hits += 1
                return cached

            self.stats.misses += 1
            if hasattr(self.inner, "create"):
                embedding = await self.inner.create(input_data)
            elif hasattr(self.inner, "embed"):
                embedding = await self.inner.embed(text)
            else:
                raise AttributeError("Inner embedder has neither 'create' nor 'embed' method")
            self.backend.set(key, embedding)
            return embedding

        # Fallback for non-single-text / token ID inputs
        if hasattr(self.inner, "create"):
            return await self.inner.create(input_data)
        raise AttributeError("Inner embedder does not support non-text inputs without 'create' method")

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        if not input_data_list:
            return []

        results: list[Optional[list[float]]] = [None] * len(input_data_list)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for idx, text in enumerate(input_data_list):
            key = self._hash_text(text)
            cached = self.backend.get(key)
            if cached is not None:
                self.stats.hits += 1
                results[idx] = cached
            else:
                self.stats.misses += 1
                miss_indices.append(idx)
                miss_texts.append(text)

        if miss_texts:
            if hasattr(self.inner, "create_batch"):
                fetched_embeddings = await self.inner.create_batch(miss_texts)
            elif hasattr(self.inner, "create"):
                fetched_embeddings = [await self.inner.create(t) for t in miss_texts]
            elif hasattr(self.inner, "embed"):
                fetched_embeddings = [await self.inner.embed(t) for t in miss_texts]
            else:
                raise AttributeError("Inner embedder missing create_batch/create/embed method")

            for idx, text, emb in zip(miss_indices, miss_texts, fetched_embeddings):
                self.backend.set(self._hash_text(text), emb)
                results[idx] = emb

        return [res for res in results if res is not None]

    async def embed(self, text: str) -> List[float]:
        """Legacy compatibility alias for create(text)."""
        key = self._hash_text(text)
        cached = self.backend.get(key)
        if cached is not None:
            self.stats.hits += 1
            return cached

        self.stats.misses += 1
        if hasattr(self.inner, "embed"):
            embedding = await self.inner.embed(text)
        elif hasattr(self.inner, "create"):
            embedding = await self.inner.create(text)
        else:
            raise AttributeError("Inner embedder has neither 'embed' nor 'create' method")
        self.backend.set(key, embedding)
        return embedding
