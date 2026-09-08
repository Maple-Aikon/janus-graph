"""Configuration loader and schemas for Janus-Graph."""

from __future__ import annotations

import json
import logging
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

logger = logging.getLogger("janus_graph.config")


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
    """LLM provider parameters.

    Default model is ``"free-ais"`` (LiteLLM proxy alias) — matches
    ``config.yaml``'s ``graphiti.llm.model`` and ensures that any code path
    reaching the field default still hits the local LiteLLM proxy rather than
    a vendor-specific Anthropic model that the proxy cannot resolve.

    See ``memory/202609/upgrade-janus-graph.md`` (2026-09-08) — defensive fix
    after the FalkorDB recovery session surfaced `'claude-3-5-sonnet-20241022'`
    strings in failed queue entries despite YAML override being correctly
    wired. The hardcode was never reached under normal config load, but we
    patch the default anyway so future daemon restarts, bare ``JanusSettings()``
    invocations, or graphiti_core internals that bypass our instance builder
    cannot leak the wrong model name.
    """
    provider: str = "openai"
    model: str = "free-ais"
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


class SearchConfig(BaseModel):
    """Knowledge-graph search & rerank configuration.

    These values are pushed into the graphiti-core SearchConfig (currently
    EDGE_HYBRID_SEARCH_MMR via ``mcp.server.search_memory``).

    Precedence (per JanusSettings):
        1. environment variable  e.g. ``JANUS_SEARCH__SIM_MIN_SCORE=0.45``
        2. YAML key              ``search.sim_min_score``
        3. class default         see below
    """
    sim_min_score: float = 0.6  # mirrors graphiti_core DEFAULT_MIN_SCORE
    mmr_lambda: float = 0.5     # mirrors graphiti_core DEFAULT_MMR_LAMBDA


class FalkorCircuitSettings(BaseModel):
    """Circuit-breaker policy wrapping FalkorDBServerManager.

    Bounded retry guard: avoids probing / restart-storming the engine on
    persistent failure. Three-state machine: CLOSED → OPEN → HALF_OPEN.

    Precedence (per JanusSettings):
        1. environment variable  e.g. ``JANUS_DAEMON__FALKOR_CIRCUIT__FAILURE_THRESHOLD=5``
        2. YAML key              ``daemon.falkor_circuit.failure_threshold``
        3. class default         see below
    """
    failure_threshold: int = 3        # consecutive failures → OPEN
    reset_timeout_sec: int = 60       # OPEN → HALF_OPEN wait
    half_open_max_probes: int = 1     # HALF_OPEN probe budget


class HTTPSettings(BaseModel):
    """aiohttp HTTP server bind settings (Phase 2 PR scope)."""
    host: str = "127.0.0.1"
    port: int = 8765


class DaemonSettings(BaseModel):
    """Phase 2/3 daemon runtime settings.

    Phase 2 fields: falkor_circuit (3-state breaker), http (aiohttp bind).
    Phase 3 fields: lock (3-mode advisory-lock for /episodes),
                    search_graph (/search/graph endpoint knobs).
    B3 hard rule: cron_loop NOT shipped in Phase 2/3 (gated by
    daemon.cron_enabled=true, deferred to Phase 4).
    """
    falkor_circuit: FalkorCircuitSettings = Field(default_factory=FalkorCircuitSettings)
    http: HTTPSettings = Field(default_factory=HTTPSettings)
    cron_enabled: bool = False  # Phase 4 gate; ignored by daemon Phase 2/3.
    # Phase 3 fields — resolved in JanusSettings.model_post_init via forward refs.
    lock: "DaemonLockSettings" = Field(default=None)  # type: ignore[assignment]
    search_graph: "DaemonSearchGraphSettings" = Field(default=None)  # type: ignore[assignment]


def _resolve_home() -> Path:
    """Resolve JANUS_GRAPH_HOME, defaulting to ~/.janus-graph/.

    All relative paths in config.yaml are resolved against this directory at
    config-load time (see ``JanusSettings.model_post_init``). The home
    directory is created lazily — ``model_post_init`` calls
    ``mkdir(parents=True, exist_ok=True)`` for any path it touches.

    Precedence:
        1. ``JANUS_GRAPH_HOME`` environment variable
        2. ``~/.janus-graph/`` (XDG-style fallback)

    Ref: goal ``janus-graph-home-env-refactor`` (2026-09-08). Paved the way
    for moving runtime state out of the source tree (``~/projects/janus-graph/``)
    into ``~/.picoclaw/workspace/apps/janus-graph/``.
    """
    env_home = os.getenv("JANUS_GRAPH_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    return (Path.home() / ".janus-graph").resolve()


def _resolve_default_yaml_file() -> Optional[Path]:
    """Locate the config YAML file.

    Precedence:
        1. ``JANUS_CONFIG_PATH`` env var (explicit override — escape hatch for
           multi-tenant / parallel test runs)
        2. ``$JANUS_GRAPH_HOME/config.yaml`` (new canonical location)
        3. ``./config.yaml`` (CWD — legacy, kept for CLI usage from project root)
        4. ``./config.example.yaml`` (last-resort dev seed)
    """
    env_path = os.getenv("JANUS_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    home = _resolve_home()
    home_cfg = home / "config.yaml"
    if home_cfg.exists():
        return home_cfg
    if Path("config.yaml").exists():
        return Path("config.yaml")
    if Path("config.example.yaml").exists():
        return Path("config.example.yaml")
    return None


# Relative-path fields that ``model_post_init`` rewrites against
# ``JANUS_GRAPH_HOME``. Anything absolute is left alone. The dotted key is
# the path inside the JanusSettings tree; the value is the type — we use the
# schema annotations to coerce back from str after rewriting.
_HOME_RELATIVE_PATH_FIELDS = (
    "engine.bin_dir",
    "engine.data_dir",
    "engine.pid_file",
    "engine.log_file",
    "pipeline.queue_db_path",
    "heuristics.quirks_log_path",
    "report.sinks.file.path",
)


def _apply_home_relative_paths(cfg: "JanusSettings", home: Path) -> None:
    """Prefix every relative path field with ``home``.

    Walks ``_HOME_RELATIVE_PATH_FIELDS`` and replaces any relative string
    with ``str(home / original_value)``. Absolute paths and ``Path`` objects
    are left untouched so operators can still pin a path to e.g.
    ``/var/lib/janus-graph`` on production hosts.

    ``dotted`` keys can be 2-4 levels deep (e.g. ``report.sinks.file.path``).
    Only the leaf is rewritten; intermediate objects keep their identity so
    any caller holding a reference to e.g. ``cfg.report.sinks.file`` still
    sees the updated ``path`` attribute.

    Called from ``JanusSettings.model_post_init`` after forward-ref defaults
    are resolved.
    """
    for dotted in _HOME_RELATIVE_PATH_FIELDS:
        parts = dotted.split(".")
        leaf = parts[-1]
        parent = cfg
        for segment in parts[:-1]:
            parent = getattr(parent, segment)
        current = getattr(parent, leaf)
        if isinstance(current, Path):
            if current.is_absolute():
                continue
            new_path = (home / current).resolve()
            setattr(parent, leaf, new_path)
            continue
        if not isinstance(current, str):
            continue
        if not current or Path(current).is_absolute():
            continue
        new_path = (home / current).resolve()
        setattr(parent, leaf, str(new_path))
    # Ensure parent dirs exist for file paths (logs/quirks/report) and the
    # SQLite parent. We deliberately do NOT mkdir() the bin_dir — that's the
    # operator's responsibility to populate. data_dir (falkordb rdb home) is
    # created so BGSAVE dump.rdb can land without churn later.
    for dotted in (
        "pipeline.queue_db_path",
        "heuristics.quirks_log_path",
        "report.sinks.file.path",
        "engine.log_file",
        "engine.data_dir",
    ):
        parts = dotted.split(".")
        leaf = parts[-1]
        parent = cfg
        for segment in parts[:-1]:
            parent = getattr(parent, segment)
        current = getattr(parent, leaf)
        if not current:
            continue
        p = Path(current)
        is_dir = dotted in ("engine.data_dir",)
        target = p if is_dir else p.parent
        target.mkdir(parents=True, exist_ok=True)


class JanusSettings(BaseSettings):
    """Main Janus-Graph Configuration Root."""
    engine: EngineConfig = Field(default_factory=EngineConfig)
    graphiti: GraphitiConfig = Field(default_factory=GraphitiConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    heuristics: HeuristicsConfig = Field(default_factory=HeuristicsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    daemon: DaemonSettings = Field(default_factory=DaemonSettings)

    model_config = SettingsConfigDict(
        env_prefix="JANUS_",
        env_nested_delimiter="__",
        extra="ignore",
        yaml_file=_resolve_default_yaml_file(),
    )

    def model_post_init(self, __context: object) -> None:
        """Resolve Phase 3 forward-ref defaults (DaemonLockSettings + DaemonSearchGraphSettings)
        + apply JANUS_GRAPH_HOME prefix to all relative paths in config.yaml.

        Forward refs are pre-bound at module import via
        ``JanusSettings.model_rebuild()`` below; this method only fills in
        None defaults that survived the rebuild.

        Home resolution happens AFTER forward-ref fills so we can read the
        fully-typed ``cfg`` tree.
        """
        from janus_graph.daemon.phase3_settings import (
            DaemonLockSettings,
            DaemonSearchGraphSettings,
        )
        if self.daemon.lock is None:
            object.__setattr__(self.daemon, "lock", DaemonLockSettings())
        if self.daemon.search_graph is None:
            object.__setattr__(self.daemon, "search_graph", DaemonSearchGraphSettings())
        # Apply JANUS_GRAPH_HOME to relative paths AFTER forward refs so the
        # schema is fully materialized and we don't have to chase lazy types.
        _apply_home_relative_paths(self, _resolve_home())

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



def resolve_search_params(cfg: JanusSettings) -> Tuple[float, float]:
    """Resolve sim_min_score / mmr_lambda with fail-soft invalid-env fallback.

    Honours the standard env > config > default precedence but, if a user
    sets an obviously-bad env value (e.g. ``"abc"``, ``"NaN"``, ``"-1"``,
    ``"1.5"``), logs a warning and falls back to the validated ``cfg.search``
    value (i.e. config-yaml → built-in default). This prevents an
    ``MCP restart + bad env`` pair from brick-ing the search_memory tool.

    Returns:
        (sim_min_score, mmr_lambda) — both in [0.0, 1.0].
    """

    def _safe(name: str, raw: Optional[str], cfg_value: float, lo: float, hi: float) -> float:
        if raw is None or raw.strip() == "":
            return cfg_value
        try:
            v = float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "search param %s=%r is not numeric; falling back to %s",
                name, raw, cfg_value,
            )
            return cfg_value
        if v != v or v < lo or v > hi:  # NaN check + range check
            logger.warning(
                "search param %s=%s out of range [%s, %s]; falling back to %s",
                name, v, lo, hi, cfg_value,
            )
            return cfg_value
        return v

    sim_raw = os.environ.get("JANUS_SEARCH__SIM_MIN_SCORE")
    mmr_raw = os.environ.get("JANUS_SEARCH__MMR_LAMBDA")
    sim = _safe("sim_min_score", sim_raw, cfg.search.sim_min_score, 0.0, 1.0)
    mmr = _safe("mmr_lambda", mmr_raw, cfg.search.mmr_lambda, 0.0, 1.0)
    return sim, mmr


# ─── Phase 3 forward-ref resolution ────────────────────────────────────
# DaemonSettings uses string forward refs for ``lock`` + ``search_graph`` so
# ``janus_graph.config`` can be imported without dragging in
# ``janus_graph.daemon.*`` (which transitively imports aiohttp — keep that
# opt-in for callers that only need non-daemon settings).
#
# Once daemon.phase3_settings is importable, bind the forward refs and rebuild
# the model so JanusSettings() resolves DaemonSettings.lock / .search_graph
# to concrete DaemonLockSettings / DaemonSearchGraphSettings instances.
def _bind_phase3_forward_refs() -> None:
    """Resolve Phase 3 forward refs on BOTH DaemonSettings and JanusSettings.

    Phase 4 fix (2026-09-07): DaemonSettings uses string forward refs for
    ``lock`` + ``search_graph`` so janus_graph.config can be imported
    without dragging in aiohttp. Both classes need model_rebuild() — when
    JanusSettings validates its ``daemon: DaemonSettings`` field, pydantic
    instantiates DaemonSettings first, which requires the forward refs to
    be resolved on DaemonSettings itself (not just on JanusSettings).
    Rebuilding only JanusSettings left ``cli sweep`` and other config-only
    callers broken (PydanticUserError: DaemonSettings is not fully defined).
    """
    try:
        from janus_graph.daemon.phase3_settings import (  # noqa: WPS433
            DaemonLockSettings,
            DaemonSearchGraphSettings,
        )
    except ImportError:  # pragma: no cover — daemon module optional
        return
    # 1. Rebuild DaemonSettings first so its forward refs are resolved.
    #    Without this, ``JanusSettings(...)`` fails when it tries to
    #    instantiate the ``daemon`` field with unresolved forward refs.
    from janus_graph.config import DaemonSettings  # local import: avoid cycle
    DaemonSettings.model_rebuild(
        _types_namespace={
            "DaemonLockSettings": DaemonLockSettings,
            "DaemonSearchGraphSettings": DaemonSearchGraphSettings,
        }
    )
    # 2. Rebuild JanusSettings so its ``daemon: DaemonSettings`` field picks
    #    up the now-resolved DaemonSettings subclass.
    JanusSettings.model_rebuild(
        _types_namespace={
            "DaemonLockSettings": DaemonLockSettings,
            "DaemonSearchGraphSettings": DaemonSearchGraphSettings,
        }
    )


_bind_phase3_forward_refs()
