"""Cache package for vector embeddings and search acceleration."""

from .base import CacheBackend
from .embed_cache import EmbedCache
from .lru import LRUCacheBackend
from .stats import CacheStats

__all__ = ["CacheBackend", "EmbedCache", "LRUCacheBackend", "CacheStats"]
