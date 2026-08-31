"""Configuration loader and schemas for Janus-Graph."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class EngineConfig(BaseModel):
    """FalkorDB / Redis engine settings."""
    host: str = "127.0.0.1"
    port: int = 6379
    bin_dir: str = "./bin"
    data_dir: str = "./data/falkordb"
    pid_file: str = "./data/falkordb/falkordb.pid"
    log_file: str = "./data/logs/falkordb.log"
    save_interval_sec: int = 900
    max_memory_mb: int = 1024


class LLMConfig(BaseModel):
    """LLM provider parameters."""
    provider: str = "openai"
    model: str = "claude-3-5-sonnet-20241022"
    base_url: str = "http://127.0.0.1:4000/v1"
    api_key: str = "sk-litellm-proxy"
    temperature: float = 0.1
    max_tokens: int = 4096


class EmbeddingConfig(BaseModel):
    """Embedding model parameters."""
    provider: str = "openai"
    model: str = "nomic-embed-text-v2"
    base_url: str = "http://127.0.0.1:8081/v1"
    api_key: str = "not-needed"
    dim: int = 768


class GraphitiConfig(BaseModel):
    """Graphiti memory configuration."""
    group_id: str = "graphiti_memory"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


class DreamConfig(BaseModel):
    """Dream mode memory consolidation settings."""
    enabled: bool = True
    force_clustering: bool = False


class PipelineConfig(BaseModel):
    """Pipeline and queue settings."""
    queue_db_path: str = "./data/episodes.db"
    worker_concurrency: int = 6
    max_attempts: int = 3
    attempt_timeout_sec: int = 120
    cron_interval_min: int = 10
    drain_batch_size: int = 50
    dream: DreamConfig = Field(default_factory=DreamConfig)


class HeuristicsConfig(BaseModel):
    """Schema repair & heuristic registry settings."""
    auto_repair: bool = True
    active_rules: List[str] = Field(
        default_factory=lambda: ["edge_duplicate", "extracted_edges", "node_resolutions"]
    )
    quirks_log_path: str = "./data/logs/llm_schema_quirks.jsonl"
    metrics_enabled: bool = True


class EmbedCacheConfig(BaseModel):
    """Embedding cache configuration."""
    enabled: bool = True
    max_size: int = 10000
    eviction: Literal["lru", "fifo"] = "lru"
    stats_interval_sec: int = 300


class CacheConfig(BaseModel):
    """Cache configuration namespace."""
    embed: EmbedCacheConfig = Field(default_factory=EmbedCacheConfig)


class FileSinkConfig(BaseModel):
    enabled: bool = True
    path: str = "./data/logs/janus_report.jsonl"
    rotation_max_bytes: int = 10485760
    backup_count: int = 5


class CLISinkConfig(BaseModel):
    enabled: bool = True
    format: Literal["pretty", "json", "summary"] = "pretty"
    min_severity: str = "warning"


class TelegramSinkConfig(BaseModel):
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""
    rate_limit_per_min: int = 10


class WebhookSinkConfig(BaseModel):
    enabled: bool = False
    url: str = ""
    secret_header: str = "X-Janus-Signature"
    secret_token: str = ""
    timeout_sec: int = 5


class ReportSinksConfig(BaseModel):
    file: FileSinkConfig = Field(default_factory=FileSinkConfig)
    cli: CLISinkConfig = Field(default_factory=CLISinkConfig)
    telegram: TelegramSinkConfig = Field(default_factory=TelegramSinkConfig)
    webhook: WebhookSinkConfig = Field(default_factory=WebhookSinkConfig)


class ReportConfig(BaseModel):
    """Multi-channel reporting configuration."""
    min_severity: str = "info"
    sinks: ReportSinksConfig = Field(default_factory=ReportSinksConfig)


def _resolve_default_yaml_file() -> Optional[Path]:
    env_path = os.getenv("JANUS_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    if Path("config.yaml").exists():
        return Path("config.yaml")
    if Path("config.example.yaml").exists():
        return Path("config.example.yaml")
    return None


class JanusSettings(BaseSettings):
    """Main Janus-Graph Configuration Root."""
    engine: EngineConfig = Field(default_factory=EngineConfig)
    graphiti: GraphitiConfig = Field(default_factory=GraphitiConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    heuristics: HeuristicsConfig = Field(default_factory=HeuristicsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    model_config = SettingsConfigDict(
        env_prefix="JANUS_",
        env_nested_delimiter="__",
        extra="ignore",
        yaml_file=_resolve_default_yaml_file(),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        yaml_file = _resolve_default_yaml_file()
        if yaml_file and yaml_file.exists():
            yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file)
            # Environment variables take precedence over YAML file settings
            return (init_settings, env_settings, yaml_settings, file_secret_settings)
        return (init_settings, env_settings, dotenv_settings, file_secret_settings)


def load_config(config_path: Optional[str | Path] = None) -> JanusSettings:
    """Load settings with fallback to YAML/JSON file and environment variable overrides."""
    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            if p.suffix in (".yaml", ".yml"):
                with open(p, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        if "graphiti" not in loaded and ("llm" in loaded or "server" in loaded or "database" in loaded):
                            from .migrate import convert_legacy_dict
                            return JanusSettings(**convert_legacy_dict(loaded))
                        return JanusSettings(**loaded)
            elif p.suffix == ".json":
                from .migrate import convert_legacy_dict
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return JanusSettings(**convert_legacy_dict(data))
    return JanusSettings()
