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


@pytest.mark.asyncio
async def test_embed_cache_embedder_client_conformance():
    from graphiti_core.embedder.client import EmbedderClient

    mock_inner = AsyncMock()
    mock_inner.create.return_value = [0.4, 0.5, 0.6]
    mock_inner.create_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]

    cache = EmbedCache(inner=mock_inner, max_size=10)
    assert isinstance(cache, EmbedderClient)

    # Test single create with string
    res = await cache.create("foo")
    assert res == [0.4, 0.5, 0.6]
    assert mock_inner.create.call_count == 1

    # Cache hit on single create with list wrapping
    res_hit = await cache.create(["foo"])
    assert res_hit == [0.4, 0.5, 0.6]
    assert mock_inner.create.call_count == 1
    assert cache.stats.hits == 1

    # Test create_batch
    batch_res = await cache.create_batch(["bar", "baz"])
    assert batch_res == [[0.1, 0.2], [0.3, 0.4]]
    assert mock_inner.create_batch.call_count == 1

    # Second batch with partial hits
    mock_inner.create_batch.return_value = [[0.9, 0.9]]
    batch_res2 = await cache.create_batch(["bar", "qux"])
    assert batch_res2 == [[0.1, 0.2], [0.9, 0.9]]
    assert mock_inner.create_batch.call_count == 2
