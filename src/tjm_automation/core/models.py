from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    resolution_width: int
    resolution_height: int
    retry_attempts: int
    retry_delay_seconds: float
    post_limit: int
    api_base_url: str
    artifacts_dir: Path
    grounding_backend: str
    grounding_timeout_seconds: float
    confidence_threshold: float
    typing_delay_ms: int
    launch_timeout_seconds: float
    window_poll_interval_seconds: float


@dataclass(frozen=True)
class OutputSettings:
    save_dir: Path
    overwrite_mode: str


@dataclass(frozen=True)
class LoggingSettings:
    level: str


@dataclass(frozen=True)
class TargetSettings:
    key: str
    display_name: str
    grounding_query: str
    window_title_contains: str
    save_dialog_title_contains: str
    double_click: bool
    filename_pattern: str
    content_template: str


@dataclass(frozen=True)
class VisionLlmSettings:
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class Settings:
    config_dir: Path
    app: AppSettings
    output: OutputSettings
    logging: LoggingSettings
    target: TargetSettings
    vision_llm: VisionLlmSettings | None = None
