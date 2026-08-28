"""Stable interface contracts for Janus-Graph.

Phase 2 (heuristics) and Phase 3 (report) code against these frozen
interfaces and dataclasses, not direct cross-module imports.

Contracts are FROZEN before parallel development.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Settings dataclasses (frozen snapshots)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueueSettings:
    sqlite_busy_timeout_ms: int = 5000
    dual_run_busy_timeout_ms: int = 5000
    dual_run_enabled: bool = False
    max_retries: int = 3
    retry_backoff_sec: float = 2.0
    batch_size: int = 20


@dataclass(frozen=True)
class DreamSettings:
    enabled: bool = True
    cron_schedule: str = "30 2 * * *"
    timeout_sec: int = 600
    community_clustering_threshold: int = 50
    orphan_pruning_enabled: bool = True
    dedup_threshold: float = 0.85


@dataclass(frozen=True)
class PipelineSettings:
    queue: QueueSettings = QueueSettings()
    dream: DreamSettings = DreamSettings()
    heuristics_active: tuple[str, ...] = ("edge_duplicate", "extracted_edges", "extracted_entities", "node_resolutions")


@dataclass(frozen=True)
class CliReportSettings:
    pipe_config_path: Path = Path("./data/pipe.conf")
    pipe_user: str = "maple"
    pipe_subprocess_method: str = "sudo"  # "sudo" only in v1.0
    pipe_sudo_args: tuple[str, ...] = ("-n",)
    pipe_timeout_sec: int = 10


@dataclass(frozen=True)
class TelegramReportSettings:
    enabled: bool = False
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    rate_limit_sec: int = 3600


@dataclass(frozen=True)
class WebhookReportSettings:
    enabled: bool = False
    url: Optional[str] = None
    secret: Optional[str] = None
    timeout_sec: int = 10


@dataclass(frozen=True)
class ReportSettings:
    sinks: tuple[str, ...] = ("file", "cli")
    file_path: Path = Path("./data/logs/reports.jsonl")
    cli: CliReportSettings = CliReportSettings()
    telegram: TelegramReportSettings = TelegramReportSettings()
    webhook: WebhookReportSettings = WebhookReportSettings()


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "openai"
    model: str = "deepseek-chat"
    api_key: str = "local"
    base_url: str = "http://127.0.0.1:4000/v1"
    temperature: float = 0.1
    timeout: float = 60.0
    structured_output_mode: str = "json_object"


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str = "openai"
    model: str = "nomic-embed-text-v2"
    api_key: str = "local"
    base_url: str = "http://127.0.0.1:8082/v1"
    dimensions: int = 768


@dataclass(frozen=True)
class DatabaseSettings:
    host: str = "127.0.0.1"
    port: int = 6379
    graph_name: str = "graphiti"
    password: Optional[str] = None
    ssl: bool = False
    timeout: float = 10.0


@dataclass(frozen=True)
class PathsSettings:
    data_dir: Path = Path("./data")
    log_dir: Path = Path("./data/logs")
    falkordb_dir: Path = Path("./data/falkordb")
    queue_db_path: Path = Path("./data/queue.db")
    quirks_log_path: Path = Path("./data/logs/llm_schema_quirks.jsonl")


@dataclass(frozen=True)
class Settings:
    """Frozen snapshot of config.yaml. Phase 2/3 must not mutate."""
    pipeline: PipelineSettings = PipelineSettings()
    report: ReportSettings = ReportSettings()
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    database: DatabaseSettings = DatabaseSettings()
    paths: PathsSettings = PathsSettings()


# ---------------------------------------------------------------------------
# Events dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepairEvent:
    """Emitted by heuristics, consumed by report. Frozen shape."""
    rule_name: str
    target_model: str
    episode_id: str
    payload_in: dict[str, Any]
    payload_out: dict[str, Any]
    success: bool
    duration_ms: float
    error_message: Optional[str] = None


@dataclass(frozen=True)
class DreamEvent:
    """Emitted by Dream Mode, consumed by report."""
    phase: str  # "clustering" | "dedup" | "prune" | "repair"
    nodes_before: int
    nodes_after: int
    duration_ms: float
    success: bool
    details: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Protocols / Interfaces
# ---------------------------------------------------------------------------

@runtime_checkable
class QuirksLogger(Protocol):
    """Phase 2 (heuristics) emits schema quirks via this interface.
    Phase 3 (report) subscribes to these events. Both must agree on shape."""

    def log_schema_quirk(
        self,
        *,
        model_name: str,
        field: str,
        observed_type: type,
        expected_type: type,
        repair_applied: bool,
        episode_id: str,
    ) -> None: ...


@runtime_checkable
class RepairRule(Protocol):
    """Pluggable repair rule. Phase 2 implements these. Phase 3 reports when fired."""
    name: str
    target_model: str
    priority: int

    def can_repair(self, schema_name: str, payload: Any, error: Optional[Exception] = None) -> bool: ...
    def repair(self, schema_name: str, payload: Any, error: Optional[Exception] = None) -> dict[str, Any]: ...


@runtime_checkable
class ReportSink(Protocol):
    """Phase 3 implements these. Phase 2 does not import directly."""
    name: str

    async def dispatch(self, event: dict[str, Any], settings: Settings) -> None: ...
