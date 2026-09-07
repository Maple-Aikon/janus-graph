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
import logging
from dataclasses import dataclass
from typing import List, Optional

import aiohttp

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
        """
        if not texts:
            return []
        if self._session is None or self._session.closed:
            await self.start()
        assert self._session is not None  # for type-checker

        async with self._semaphore:
            url = f"{self._settings.embedding_base_url.rstrip('/')}/embeddings"
            payload = {
                "model": self._settings.embedding_model,
                "input": texts,
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
                        return [EmbeddingResult(t, None, err) for t in texts]
                    data = await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning("embed transport error: %s", e)
                return [EmbeddingResult(t, None, str(e)) for t in texts]

        # Normalize response: {"data": [{"embedding": [...]}, ...]}
        embeddings_raw = data.get("data") or []
        results: List[EmbeddingResult] = []
        for idx, text in enumerate(texts):
            if idx < len(embeddings_raw) and isinstance(embeddings_raw[idx], dict):
                emb = embeddings_raw[idx].get("embedding")
                if isinstance(emb, list):
                    results.append(EmbeddingResult(text, emb, None))
                    continue
            results.append(EmbeddingResult(text, None, "missing_embedding_in_response"))
        return results
