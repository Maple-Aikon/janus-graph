"""Async embedding client for /search/graph rerank (Phase 3.1).

Talks to an OpenAI-compatible ``/embeddings`` endpoint (default:
``http://127.0.0.1:8081/v1`` — the local nomic-embed service started by
PicoClaw). Concurrency bounded by a semaphore (default 4) so a runaway
batch can't OOM the daemon or starve /health.

Fail-soft: connection refused / 5xx returns empty list; SearchEngine flips
``degraded=True`` in that case. We do NOT raise — the caller treats the
BFS result as unranked and returns raw rows with cosine=None.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional

import aiohttp

from ..cache.lru import LRUCacheBackend
from ..cache.stats import CacheStats
from .phase3_settings import DaemonSearchGraphSettings

logger = logging.getLogger("janus_graph.daemon.embedding_client")


@dataclass
class EmbeddingResult:
    """Per-text embedding result (or failure sentinel)."""

    text: str
    embedding: Optional[List[float]]
    error: Optional[str] = None


class EmbeddingClient:
    """Stateless async embedding client.

    Lifecycle:
      - ``await client.start()`` — opens aiohttp ClientSession (idempotent).
      - ``await client.embed(texts)`` — returns List[EmbeddingResult].
      - ``await client.stop()`` — closes the session (called on shutdown).

    Concurrency: each ``embed()`` call batches all ``texts`` into one HTTP
    request (most OpenAI-compatible servers accept array input), but we
    still gate with a semaphore in case callers fire multiple embed calls
    concurrently from different request handlers.
    """

    def __init__(self, settings: DaemonSearchGraphSettings):
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.embedding_concurrency)
        self._session: Optional[aiohttp.ClientSession] = None
        # Per-text embedding cache (2026-10-04). Reuses the same LRU backend
        # and CacheStats primitives as ``janus_graph.cache.EmbedCache``, but
        # lives HERE rather than wrapping that class: EmbedCache exposes
        # ``embed(text: str) -> vector`` (graphiti EmbedderClient shape) while
        # this client must keep ``embed(texts: List[str]) -> List[
        # EmbeddingResult]`` (batch + per-text failure sentinel). Wrapping it
        # would make the batch path 1 HTTP request per text.
        self._cache = LRUCacheBackend(max_size=settings.embed_cache_size)
        self.cache_stats = CacheStats()

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def start(self) -> None:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10.0)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def stop(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def embed(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed ``texts`` via OpenAI-compatible /embeddings endpoint.

        Returns one ``EmbeddingResult`` per input (positional order preserved).
        On transport failure returns the whole batch as
        ``EmbeddingResult(text, None, "<error>")`` so the caller can fall
        back to BFS-only results without crashing.

        Served from the in-process LRU where possible: only texts that miss
        (and did not previously fail) reach the HTTP endpoint, in ONE batched
        request. Failures are never cached -- a transient timeout must not
        poison the entry for the rest of the daemon's life.
        """
        if not texts:
            return []
        if self._session is None or self._session.closed:
            await self.start()
        assert self._session is not None  # for type-checker

        # ── cache lookup ──────────────────────────────────────────────
        results: List[Optional[EmbeddingResult]] = [None] * len(texts)
        miss_texts: List[str] = []
        miss_index: List[int] = []
        for idx, text in enumerate(texts):
            cached = self._cache.get(self._cache_key(text))
            if cached is not None:
                self.cache_stats.hits += 1
                results[idx] = EmbeddingResult(text, cached, None)
            else:
                self.cache_stats.misses += 1
                miss_texts.append(text)
                miss_index.append(idx)

        if not miss_texts:
            return [r if r is not None else EmbeddingResult(t, None, "not_cached")
                    for r, t in zip(results, texts)]

        async with self._semaphore:
            url = f"{self._settings.embedding_base_url.rstrip('/')}/embeddings"
            payload = {
                "model": self._settings.embedding_model,
                # Only the misses go on the wire. Cache hits must NOT be
                # re-sent -- that would keep the 10s timeout alive on the
                # exact inputs the cache exists to remove.
                "input": miss_texts,
            }
            headers = {
                "Authorization": f"Bearer {self._settings.embedding_api_key}",
                "Content-Type": "application/json",
            }
            try:
                async with self._session.post(url, json=payload, headers=headers) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        err = f"http_{resp.status}: {body[:200]}"
                        logger.warning("embed failed: %s", err)
                        return [r if r is not None else EmbeddingResult(t, None, err)
                                for r, t in zip(results, texts)]
                    data = await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning("embed transport error: %s", e)
                return [r if r is not None else EmbeddingResult(t, None, str(e))
                        for r, t in zip(results, texts)]

        # Normalize response: {"data": [{"embedding": [...]}, ...]}
        # Positions align with ``miss_texts`` (the batch we actually sent),
        # NOT with ``texts`` -- cache hits shift everything down.
        embeddings_raw = data.get("data") or []
        for pos, idx in enumerate(miss_index):
            text = miss_texts[pos]
            emb = None
            if pos < len(embeddings_raw) and isinstance(embeddings_raw[pos], dict):
                cand = embeddings_raw[pos].get("embedding")
                if isinstance(cand, list):
                    emb = cand
            if emb is not None:
                # Only successful vectors are cached; failures must retry.
                self._cache.set(self._cache_key(text), emb)
                results[idx] = EmbeddingResult(text, emb, None)
            else:
                results[idx] = EmbeddingResult(text, None, "missing_embedding_in_response")

        return [r if r is not None else EmbeddingResult(t, None, "not_cached")
                for r, t in zip(results, texts)]
