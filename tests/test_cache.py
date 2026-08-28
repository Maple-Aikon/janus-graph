"""Tests for LRU cache backend and EmbedCache."""

import pytest
from unittest.mock import AsyncMock
from janus_graph.cache.lru import LRUCacheBackend
from janus_graph.cache.embed_cache import EmbedCache


def test_lru_cache_eviction():
    cache = LRUCacheBackend(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    cache.set("c", 3)  # should evict "b" because "a" was accessed
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


@pytest.mark.asyncio
async def test_embed_cache():
    mock_inner = AsyncMock()
    mock_inner.embed.return_value = [0.1, 0.2, 0.3]

    cache = EmbedCache(inner=mock_inner, max_size=10)
    res1 = await cache.embed("hello world")
    assert res1 == [0.1, 0.2, 0.3]
    assert mock_inner.embed.call_count == 1
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0

    # Repeat call should hit cache
    res2 = await cache.embed("hello world")
    assert res2 == [0.1, 0.2, 0.3]
    assert mock_inner.embed.call_count == 1
    assert cache.stats.hits == 1
    assert cache.stats.hit_rate == 0.5
