"""Graphiti instance builder with schema repair and embed cache integration."""

from __future__ import annotations

from typing import Any, Optional
from ..config import JanusSettings, load_config


def create_graphiti_instance(settings: Optional[JanusSettings] = None) -> Any:
    """Instantiate and configure Graphiti memory engine with repair client."""
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EntityNode
    from graphiti_core.edges import EntityEdge
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.llm_client.config import LLMConfig as _CoreLLMConfig
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from ..heuristics.repairing_client import SchemaRepairingLLMClient
    from ..cache.embed_cache import EmbedCache

    cfg = settings or load_config()

    # graphiti_core >=0.20 OpenAIGenericClient signature:
    #   (config: LLMConfig | None, cache=False, client=None, max_tokens=..., structured_output_mode='json_schema')
    # The legacy kwargs (model=, api_key=, base_url=, temperature=) no longer exist.
    base_llm = OpenAIGenericClient(
        config=_CoreLLMConfig(
            api_key=cfg.graphiti.llm.api_key,
            model=cfg.graphiti.llm.model,
            base_url=cfg.graphiti.llm.base_url,
            temperature=cfg.graphiti.llm.temperature,
            max_tokens=cfg.graphiti.llm.max_tokens,
        ),
        cache=False,
        structured_output_mode="json_object",  # LiteLLM proxy: schema-injected mode
    )

    llm_client = SchemaRepairingLLMClient(
        inner=base_llm,
        active_rules=cfg.heuristics.active_rules if cfg.heuristics.auto_repair else [],
        quirks_log_path=cfg.heuristics.quirks_log_path,
    )

    # graphiti_core OpenAIEmbedder signature:
    #   (config: OpenAIEmbedderConfig | None, client=None) — no legacy model=/api_key=/base_url= kwargs.
    base_embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=cfg.graphiti.embedding.api_key,
            base_url=cfg.graphiti.embedding.base_url,
            embedding_model=cfg.graphiti.embedding.model,
            embedding_dim=cfg.graphiti.embedding.dim,
        )
    )

    if cfg.cache.embed.enabled:
        embedder: Any = EmbedCache(
            inner=base_embedder,
            max_size=cfg.cache.embed.max_size,
        )
    else:
        embedder = base_embedder

    # graphiti_core >=0.20 Graphiti.__init__ signature:
    #   (uri, user, password, llm_client, embedder, cross_encoder,
    #    store_raw_episode_content, graph_driver, max_coroutines, tracer, trace_span_prefix)
    # The legacy falkordb_host= / falkordb_port= kwargs were removed; pass a FalkorDriver instead.
    graph_database = (
        cfg.database.graph_name
        if hasattr(cfg, "database") and getattr(cfg.database, "graph_name", None)
        else "graphiti_memory"
    )
    graph_driver = FalkorDriver(
        host=cfg.engine.host,
        port=cfg.engine.port,
        database=graph_database,
    )

    cross_encoder = OpenAIRerankerClient(
        config=_CoreLLMConfig(
            api_key=cfg.graphiti.llm.api_key,
            base_url=cfg.graphiti.llm.base_url,
        )
    )

    return Graphiti(
        graph_driver=graph_driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )
