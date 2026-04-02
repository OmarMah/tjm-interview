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
    api_timeout_seconds: float = 15.0
    close_timeout_seconds: float = 10.0
    show_desktop_delay_seconds: float = 0.4
    focus_delay_seconds: float = 0.1
    notepad_focus_delay_seconds: float = 0.2
    popup_timeout_seconds: float = 0.4
    popup_poll_interval_seconds: float = 0.1
    popup_action_delay_seconds: float = 0.2
    click_interval_ms: int = 120
    notepad_select_all_delay_seconds: float = 0.1
    notepad_content_paste_delay_seconds: float = 0.2
    notepad_save_shortcut_delay_seconds: float = 0.3
    notepad_save_dialog_focus_delay_seconds: float = 0.2
    notepad_save_field_delay_seconds: float = 0.1
    notepad_save_submit_delay_seconds: float = 0.1


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
