from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from tjm_automation.core.errors import ConfigurationError
from tjm_automation.core.models import (
    AppSettings,
    LoggingSettings,
    OutputSettings,
    Settings,
    TargetSettings,
    VisionLlmSettings,
)


CONFIG_ENV_VAR = "TJM_CONFIG_DIR"
VISION_LLM_BASE_URL_ENV_VAR = "TJM_VLM_BASE_URL"
VISION_LLM_API_KEY_ENV_VAR = "TJM_VLM_API_KEY"
VISION_LLM_MODEL_ENV_VAR = "TJM_VLM_MODEL"
VISION_LLM_BACKENDS = {"vision_llm", "vlm", "openai_compatible"}


def load_settings(config_dir: Path | None = None, target_name: str = "notepad") -> Settings:
    resolved_config_dir = resolve_config_dir(config_dir)
    project_root = resolved_config_dir.parent
    _load_dotenv_file(project_root / ".env")

    app_config = _load_toml(resolved_config_dir / "app.toml")
    target_config = _load_toml(resolved_config_dir / "targets" / f"{target_name}.toml")

    app_section = _require_section(app_config, "app", source="app.toml")
    output_section = _require_section(app_config, "output", source="app.toml")
    logging_section = app_config.get("logging", {})
    target_section = _require_section(target_config, "target", source=f"{target_name}.toml")

    grounding_backend = _require_str(app_section, "grounding_backend", "app.toml")
    settings = Settings(
        config_dir=resolved_config_dir,
        app=AppSettings(
            resolution_width=_require_int(app_section, "resolution_width", "app.toml"),
            resolution_height=_require_int(app_section, "resolution_height", "app.toml"),
            retry_attempts=_require_int(app_section, "retry_attempts", "app.toml"),
            retry_delay_seconds=_require_float(app_section, "retry_delay_seconds", "app.toml"),
            post_limit=_require_int(app_section, "post_limit", "app.toml"),
            api_base_url=_require_str(app_section, "api_base_url", "app.toml"),
            api_timeout_seconds=_optional_float(
                app_section,
                "api_timeout_seconds",
                default=15.0,
            ),
            artifacts_dir=_resolve_path(
                _require_str(app_section, "artifacts_dir", "app.toml"),
                base_dir=project_root,
            ),
            grounding_backend=grounding_backend,
            grounding_timeout_seconds=_optional_float(
                app_section,
                "grounding_timeout_seconds",
                default=60.0,
            ),
            confidence_threshold=_require_float(app_section, "confidence_threshold", "app.toml"),
            typing_delay_ms=_require_int(app_section, "typing_delay_ms", "app.toml"),
            launch_timeout_seconds=_optional_float(
                app_section,
                "launch_timeout_seconds",
                default=8.0,
            ),
            close_timeout_seconds=_optional_float(
                app_section,
                "close_timeout_seconds",
                default=10.0,
            ),
            window_poll_interval_seconds=_optional_float(
                app_section,
                "window_poll_interval_seconds",
                default=0.1,
            ),
            show_desktop_delay_seconds=_optional_float(
                app_section,
                "show_desktop_delay_seconds",
                default=0.4,
            ),
            focus_delay_seconds=_optional_float(
                app_section,
                "focus_delay_seconds",
                default=0.1,
            ),
            notepad_focus_delay_seconds=_optional_float(
                app_section,
                "notepad_focus_delay_seconds",
                default=0.2,
            ),
            popup_timeout_seconds=_optional_float(
                app_section,
                "popup_timeout_seconds",
                default=0.4,
            ),
            popup_poll_interval_seconds=_optional_float(
                app_section,
                "popup_poll_interval_seconds",
                default=0.1,
            ),
            popup_action_delay_seconds=_optional_float(
                app_section,
                "popup_action_delay_seconds",
                default=0.2,
            ),
            click_interval_ms=_optional_int(
                app_section,
                "click_interval_ms",
                default=120,
            ),
            notepad_select_all_delay_seconds=_optional_float(
                app_section,
                "notepad_select_all_delay_seconds",
                default=0.1,
            ),
            notepad_content_paste_delay_seconds=_optional_float(
                app_section,
                "notepad_content_paste_delay_seconds",
                default=0.2,
            ),
            notepad_save_shortcut_delay_seconds=_optional_float(
                app_section,
                "notepad_save_shortcut_delay_seconds",
                default=0.3,
            ),
            notepad_save_dialog_focus_delay_seconds=_optional_float(
                app_section,
                "notepad_save_dialog_focus_delay_seconds",
                default=0.2,
            ),
            notepad_save_field_delay_seconds=_optional_float(
                app_section,
                "notepad_save_field_delay_seconds",
                default=0.1,
            ),
            notepad_save_submit_delay_seconds=_optional_float(
                app_section,
                "notepad_save_submit_delay_seconds",
                default=0.1,
            ),
        ),
        output=OutputSettings(
            save_dir=_resolve_path(
                _require_str(output_section, "save_dir", "app.toml"),
                base_dir=project_root,
            ),
            overwrite_mode=_require_str(output_section, "overwrite_mode", "app.toml"),
        ),
        logging=LoggingSettings(
            level=_optional_str(logging_section, "level", default="INFO"),
        ),
        target=TargetSettings(
            key=_require_str(target_section, "key", f"{target_name}.toml"),
            display_name=_require_str(target_section, "display_name", f"{target_name}.toml"),
            grounding_query=_require_str(target_section, "grounding_query", f"{target_name}.toml"),
            window_title_contains=_require_str(
                target_section,
                "window_title_contains",
                f"{target_name}.toml",
            ),
            save_dialog_title_contains=_require_str(
                target_section,
                "save_dialog_title_contains",
                f"{target_name}.toml",
            ),
            double_click=_require_bool(target_section, "double_click", f"{target_name}.toml"),
            filename_pattern=_require_str(target_section, "filename_pattern", f"{target_name}.toml"),
            content_template=_require_str(target_section, "content_template", f"{target_name}.toml"),
        ),
        vision_llm=_load_vision_llm_settings(grounding_backend),
    )
    _validate_settings(settings)
    return settings


def resolve_config_dir(config_dir: Path | None = None) -> Path:
    if config_dir is not None:
        candidate = Path(config_dir).expanduser().resolve()
        if not candidate.exists():
            raise ConfigurationError(f"Config directory does not exist: {candidate}")
        return candidate

    env_value = os.getenv(CONFIG_ENV_VAR)
    if env_value:
        candidate = Path(os.path.expandvars(env_value)).expanduser().resolve()
        if not candidate.exists():
            raise ConfigurationError(
                f"Environment variable {CONFIG_ENV_VAR} points to a missing directory: {candidate}"
            )
        return candidate

    for base in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        candidate = base / "config"
        if (candidate / "app.toml").exists():
            return candidate

    raise ConfigurationError(
        "Unable to locate config directory. Pass --config-dir or set TJM_CONFIG_DIR."
    )


def _load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Required config file is missing: {path}")
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise ConfigurationError(f"Config file did not parse into a table: {path}")
    return data


def _load_vision_llm_settings(grounding_backend: str) -> VisionLlmSettings | None:
    if grounding_backend.strip().casefold() not in VISION_LLM_BACKENDS:
        return None

    return VisionLlmSettings(
        base_url=_require_env(VISION_LLM_BASE_URL_ENV_VAR),
        api_key=_require_env(VISION_LLM_API_KEY_ENV_VAR),
        model=_require_env(VISION_LLM_MODEL_ENV_VAR),
    )


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(
            f"Missing required environment variable '{name}' for the vision LLM backend."
        )
    return value


def _require_section(data: dict[str, Any], key: str, *, source: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Missing [{key}] section in {source}")
    return value


def _require_str(data: dict[str, Any], key: str, source: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Expected a non-empty string for '{key}' in {source}")
    return value


def _optional_str(data: dict[str, Any], key: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Expected a non-empty string for '{key}'")
    return value


def _require_int(data: dict[str, Any], key: str, source: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"Expected an integer for '{key}' in {source}")
    return value


def _require_float(data: dict[str, Any], key: str, source: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"Expected a number for '{key}' in {source}")
    return float(value)


def _optional_float(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"Expected a number for '{key}'")
    return float(value)


def _optional_int(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"Expected an integer for '{key}'")
    return value


def _require_bool(data: dict[str, Any], key: str, source: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ConfigurationError(f"Expected a boolean for '{key}' in {source}")
    return value


def _resolve_path(value: str, *, base_dir: Path) -> Path:
    expanded = Path(os.path.expandvars(value)).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (base_dir / expanded).resolve()


def _validate_settings(settings: Settings) -> None:
    if settings.app.resolution_width <= 0 or settings.app.resolution_height <= 0:
        raise ConfigurationError("Screen resolution must be positive.")
    if settings.app.retry_attempts < 1:
        raise ConfigurationError("retry_attempts must be at least 1.")
    if settings.app.retry_delay_seconds < 0:
        raise ConfigurationError("retry_delay_seconds must be non-negative.")
    if settings.app.post_limit < 1:
        raise ConfigurationError("post_limit must be at least 1.")
    if settings.app.api_timeout_seconds <= 0:
        raise ConfigurationError("api_timeout_seconds must be positive.")
    if settings.app.grounding_timeout_seconds <= 0:
        raise ConfigurationError("grounding_timeout_seconds must be positive.")
    if not 0 <= settings.app.confidence_threshold <= 1:
        raise ConfigurationError("confidence_threshold must be between 0 and 1.")
    if settings.app.typing_delay_ms < 0:
        raise ConfigurationError("typing_delay_ms must be non-negative.")
    if settings.app.launch_timeout_seconds <= 0:
        raise ConfigurationError("launch_timeout_seconds must be positive.")
    if settings.app.close_timeout_seconds <= 0:
        raise ConfigurationError("close_timeout_seconds must be positive.")
    if settings.app.window_poll_interval_seconds <= 0:
        raise ConfigurationError("window_poll_interval_seconds must be positive.")
    if settings.app.show_desktop_delay_seconds < 0:
        raise ConfigurationError("show_desktop_delay_seconds must be non-negative.")
    if settings.app.focus_delay_seconds < 0:
        raise ConfigurationError("focus_delay_seconds must be non-negative.")
    if settings.app.notepad_focus_delay_seconds < 0:
        raise ConfigurationError("notepad_focus_delay_seconds must be non-negative.")
    if settings.app.popup_timeout_seconds < 0:
        raise ConfigurationError("popup_timeout_seconds must be non-negative.")
    if settings.app.popup_poll_interval_seconds <= 0:
        raise ConfigurationError("popup_poll_interval_seconds must be positive.")
    if settings.app.popup_action_delay_seconds < 0:
        raise ConfigurationError("popup_action_delay_seconds must be non-negative.")
    if settings.app.click_interval_ms < 0:
        raise ConfigurationError("click_interval_ms must be non-negative.")
    if settings.app.notepad_select_all_delay_seconds < 0:
        raise ConfigurationError("notepad_select_all_delay_seconds must be non-negative.")
    if settings.app.notepad_content_paste_delay_seconds < 0:
        raise ConfigurationError("notepad_content_paste_delay_seconds must be non-negative.")
    if settings.app.notepad_save_shortcut_delay_seconds < 0:
        raise ConfigurationError("notepad_save_shortcut_delay_seconds must be non-negative.")
    if settings.app.notepad_save_dialog_focus_delay_seconds < 0:
        raise ConfigurationError("notepad_save_dialog_focus_delay_seconds must be non-negative.")
    if settings.app.notepad_save_field_delay_seconds < 0:
        raise ConfigurationError("notepad_save_field_delay_seconds must be non-negative.")
    if settings.app.notepad_save_submit_delay_seconds < 0:
        raise ConfigurationError("notepad_save_submit_delay_seconds must be non-negative.")
    if settings.output.overwrite_mode not in {"skip", "overwrite", "fail"}:
        raise ConfigurationError("overwrite_mode must be one of: skip, overwrite, fail.")
