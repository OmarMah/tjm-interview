"""Thin Notepad target adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import sleep

from tjm_automation.automation.desktop import Point, WindowInfo, WindowsDesktopController
from tjm_automation.core.errors import AutomationError, TargetError
from tjm_automation.core.models import Settings
from tjm_automation.grounding import GroundingPipeline, build_grounding_pipeline
from tjm_automation.integrations.storage import prepare_output_path, wait_for_path


@dataclass(frozen=True)
class SavedDocumentResult:
    path: Path
    status: str


class NotepadTargetAdapter:
    """Launches the configured Notepad target from its desktop icon."""

    def __init__(
        self,
        settings: Settings,
        *,
        desktop_controller: WindowsDesktopController | None = None,
        grounding_pipeline: GroundingPipeline | None = None,
    ) -> None:
        self._settings = settings
        self._desktop = desktop_controller or WindowsDesktopController(
            show_desktop_delay_seconds=settings.app.show_desktop_delay_seconds,
            click_interval_ms=settings.app.click_interval_ms,
            focus_delay_seconds=settings.app.focus_delay_seconds,
            popup_poll_interval_seconds=settings.app.popup_poll_interval_seconds,
            popup_action_delay_seconds=settings.app.popup_action_delay_seconds,
        )
        self._grounding = grounding_pipeline or build_grounding_pipeline(settings)
        self._cached_launch_point: Point | None = None

    def launch_from_desktop(self) -> WindowInfo:
        existing_window = self._find_existing_window()
        if existing_window is not None:
            return existing_window

        self._desktop.show_desktop()
        cached_point = self._load_cached_launch_point()
        if cached_point is not None:
            try:
                return self._launch_from_point(cached_point)
            except AutomationError:
                self._desktop.show_desktop()

        point = self._locate_launch_point()
        window = self._launch_from_point(point)
        self._store_cached_launch_point(point)
        return window

    def _find_existing_window(self) -> WindowInfo | None:
        matches = self._desktop.find_windows_by_title(self._settings.target.window_title_contains)
        if not matches:
            return None
        return next((match for match in matches if match.is_foreground), matches[0])

    def _locate_launch_point(self) -> Point:
        target = self._settings.target
        screenshot = self._desktop.capture_screenshot(
            self._desktop.build_screenshot_path(
                self._settings.app.artifacts_dir,
                prefix=f"{target.key}_launch",
            )
        )
        self._validate_screenshot_dimensions(screenshot)

        detection = self._grounding.locate(
            screenshot_path=screenshot.path,
            target=target,
            artifacts_dir=self._settings.app.artifacts_dir,
        )
        return Point(x=detection.center_x, y=detection.center_y)

    def _launch_from_point(self, point: Point) -> WindowInfo:
        target = self._settings.target
        if target.double_click:
            self._desktop.double_click(point)
        else:
            self._desktop.click(point)
        return self._desktop.wait_for_window(
            target.window_title_contains,
            timeout_seconds=self._settings.app.launch_timeout_seconds,
            poll_interval_seconds=self._settings.app.window_poll_interval_seconds,
        )

    def write_text_file(self, *, content: str, filename: str) -> SavedDocumentResult:
        prepared_path = prepare_output_path(
            base_dir=self._settings.output.save_dir,
            filename=filename,
            overwrite_mode=self._settings.output.overwrite_mode,
        )
        if not prepared_path.should_write:
            return SavedDocumentResult(path=prepared_path.path, status="skipped")

        window = self.launch_from_desktop()
        self._focus_window(window.handle)
        self._dismiss_process_popup(window.handle)
        self._desktop.send_hotkey("ctrl", "a")
        sleep(self._settings.app.notepad_select_all_delay_seconds)
        self._desktop.send_text(content, interval_ms=self._settings.app.typing_delay_ms)
        sleep(self._settings.app.notepad_content_paste_delay_seconds)
        self._save_current_document(prepared_path.path, document_handle=window.handle)
        saved_path = wait_for_path(
            prepared_path.path,
            timeout_seconds=self._settings.app.launch_timeout_seconds,
            poll_interval_seconds=self._settings.app.window_poll_interval_seconds,
        )
        self._close_document_window(window.handle)
        return SavedDocumentResult(path=saved_path, status="saved")

    def _save_current_document(self, output_path: Path, *, document_handle: int) -> None:
        self._focus_window(document_handle)
        self._dismiss_process_popup(document_handle)
        self._desktop.send_hotkey("ctrl", "shift", "s")
        sleep(self._settings.app.notepad_save_shortcut_delay_seconds)
        self._dismiss_process_popup(document_handle)
        save_dialog = self._desktop.wait_for_window(
            self._settings.target.save_dialog_title_contains,
            timeout_seconds=self._settings.app.launch_timeout_seconds,
            poll_interval_seconds=self._settings.app.window_poll_interval_seconds,
        )
        self._desktop.focus_window(save_dialog.handle)
        sleep(self._settings.app.notepad_save_dialog_focus_delay_seconds)
        self._desktop.send_hotkey("alt", "n")
        sleep(self._settings.app.notepad_save_field_delay_seconds)
        self._desktop.send_hotkey("ctrl", "a")
        sleep(self._settings.app.notepad_save_field_delay_seconds)
        self._desktop.send_text(str(output_path), interval_ms=self._settings.app.typing_delay_ms)
        sleep(self._settings.app.notepad_save_submit_delay_seconds)
        self._desktop.send_hotkey("alt", "s")
        self._desktop.wait_for_window_closed(
            save_dialog.handle,
            timeout_seconds=self._settings.app.launch_timeout_seconds,
            poll_interval_seconds=self._settings.app.window_poll_interval_seconds,
        )

    def _close_document_window(self, handle: int) -> None:
        self._focus_window(handle)
        self._dismiss_process_popup(handle)
        self._desktop.close_window(handle)
        self._desktop.wait_for_window_closed(
            handle,
            timeout_seconds=self._settings.app.close_timeout_seconds,
            poll_interval_seconds=self._settings.app.window_poll_interval_seconds,
        )

    def _dismiss_process_popup(self, document_handle: int) -> None:
        self._desktop.dismiss_process_popup(
            document_handle,
            timeout_seconds=self._settings.app.popup_timeout_seconds,
        )
        self._focus_window(document_handle)

    def _focus_window(self, handle: int) -> None:
        self._desktop.focus_window(handle)
        sleep(self._settings.app.notepad_focus_delay_seconds)

    def _validate_screenshot_dimensions(self, screenshot) -> None:
        expected_width = self._settings.app.resolution_width
        expected_height = self._settings.app.resolution_height
        if (screenshot.width, screenshot.height) == (expected_width, expected_height):
            return

        raise TargetError(
            f"Screenshot dimensions {screenshot.width}x{screenshot.height} did not match the "
            f"configured resolution {expected_width}x{expected_height}."
        )

    def _load_cached_launch_point(self) -> Point | None:
        if self._cached_launch_point is not None:
            return self._cached_launch_point

        cache_path = self._launch_cache_path()
        if not cache_path.exists():
            return None

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        expected_width = self._settings.app.resolution_width
        expected_height = self._settings.app.resolution_height
        if (
            payload.get("resolution_width") != expected_width
            or payload.get("resolution_height") != expected_height
        ):
            return None

        x = payload.get("x")
        y = payload.get("y")
        if not isinstance(x, int) or not isinstance(y, int):
            return None
        if not (0 <= x < expected_width and 0 <= y < expected_height):
            return None

        self._cached_launch_point = Point(x=x, y=y)
        return self._cached_launch_point

    def _store_cached_launch_point(self, point: Point) -> None:
        self._cached_launch_point = point
        cache_path = self._launch_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "x": point.x,
            "y": point.y,
            "resolution_width": self._settings.app.resolution_width,
            "resolution_height": self._settings.app.resolution_height,
        }
        try:
            cache_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        except OSError:
            pass

    def _launch_cache_path(self) -> Path:
        return self._settings.app.artifacts_dir / "cache" / f"{self._settings.target.key}_launch_point.json"


__all__ = ["NotepadTargetAdapter", "SavedDocumentResult"]
