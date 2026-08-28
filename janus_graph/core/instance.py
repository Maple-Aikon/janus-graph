"""Graphiti instance builder with schema repair and embed cache integration."""

from __future__ import annotations

from typing import Any, Optional
from ..config import JanusSettings, load_config


def create_graphiti_instance(settings: Optional[JanusSettings] = None) -> Any:
    """Instantiate and configure Graphiti memory engine with repair client."""
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EntityNode
    from graphiti_core.edges import EntityEdge
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.embedder.openai import OpenAIEmbedder
    from ..heuristics.repairing_client import SchemaRepairingLLMClient
    from ..cache.embed_cache import EmbedCache

    cfg = settings or load_config()

    base_llm = OpenAIGenericClient(
        model=cfg.graphiti.llm.model,
        api_key=cfg.graphiti.llm.api_key,
        base_url=cfg.graphiti.llm.base_url,
        temperature=cfg.graphiti.llm.temperature,
    )

    llm_client = SchemaRepairingLLMClient(
        inner=base_llm,
        active_rules=cfg.heuristics.active_rules if cfg.heuristics.auto_repair else [],
        quirks_log_path=cfg.heuristics.quirks_log_path,
    )

    base_embedder = OpenAIEmbedder(
        model=cfg.graphiti.embedding.model,
        api_key=cfg.graphiti.embedding.api_key,
        base_url=cfg.graphiti.embedding.base_url,
    )

    if cfg.cache.embed.enabled:
        embedder: Any = EmbedCache(
            inner=base_embedder,
            max_size=cfg.cache.embed.max_size,
        )
    else:
        embedder = base_embedder

    return Graphiti(
        falkordb_host=cfg.engine.host,
        falkordb_port=cfg.engine.port,
        llm_client=llm_client,
        embedder=embedder,
    )
