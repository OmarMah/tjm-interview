from __future__ import annotations

import ctypes
import subprocess
import sys
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key
from pynput.mouse import Button, Controller as MouseController

from tjm_automation.core.errors import AutomationError


SW_RESTORE = 9
WM_CLOSE = 0x0010
POPUP_BUTTON_LABELS = ("ok", "close", "exit")
POPUP_BUTTON_PRIORITY = {label: index for index, label in enumerate(POPUP_BUTTON_LABELS)}
FORM_CONTROL_TYPES = {
    "ComboBox",
    "DataGrid",
    "DataItem",
    "Document",
    "Edit",
    "Header",
    "List",
    "ListItem",
    "Table",
    "ToolBar",
    "Tree",
    "TreeItem",
}


if sys.platform == "win32":
    USER32 = ctypes.WinDLL("user32", use_last_error=True)
else:
    USER32 = None


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class WindowInfo:
    handle: int
    title: str
    is_foreground: bool


@dataclass(frozen=True)
class ScreenshotResult:
    path: Path
    width: int
    height: int


class WindowsDesktopController:
    """Windows-only desktop automation helpers backed by pynput input control."""

    def __init__(
        self,
        *,
        powershell_executable: str = "powershell",
        show_desktop_delay_seconds: float = 0.4,
        click_interval_ms: int = 120,
        focus_delay_seconds: float = 0.1,
        popup_poll_interval_seconds: float = 0.1,
        popup_action_delay_seconds: float = 0.2,
    ) -> None:
        _ensure_windows()
        self._user32 = USER32
        self._powershell_executable = powershell_executable
        self._show_desktop_delay_seconds = show_desktop_delay_seconds
        self._click_interval_ms = click_interval_ms
        self._focus_delay_seconds = focus_delay_seconds
        self._popup_poll_interval_seconds = popup_poll_interval_seconds
        self._popup_action_delay_seconds = popup_action_delay_seconds
        self._mouse = MouseController()
        self._keyboard = KeyboardController()
        _make_process_dpi_aware(self._user32)

    def build_screenshot_path(self, artifacts_dir: Path, prefix: str = "desktop") -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return artifacts_dir / "screenshots" / f"{prefix}_{stamp}.png"

    def capture_screenshot(self, output_path: Path) -> ScreenshotResult:
        destination = output_path.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                self._powershell_executable,
                "-NoProfile",
                "-Command",
                build_screenshot_script(destination),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise AutomationError(f"Failed to capture screenshot: {stderr}")

        width, height = read_png_dimensions(destination)
        return ScreenshotResult(
            path=destination,
            width=width,
            height=height,
        )

    def show_desktop(self) -> None:
        result = subprocess.run(
            [
                self._powershell_executable,
                "-NoProfile",
                "-Command",
                "(New-Object -ComObject Shell.Application).MinimizeAll()",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            self.send_hotkey("win", "d")
        sleep(self._show_desktop_delay_seconds)

    def move_mouse(self, point: Point) -> None:
        self._mouse.position = (point.x, point.y)

    def click(
        self,
        point: Point,
        *,
        button: str = "left",
        clicks: int = 1,
        interval_ms: int | None = None,
    ) -> None:
        if clicks < 1:
            raise AutomationError("clicks must be at least 1.")
        self.move_mouse(point)

        resolved_interval_ms = self._click_interval_ms if interval_ms is None else interval_ms
        resolved_button = resolve_mouse_button(button)
        for index in range(clicks):
            self._mouse.press(resolved_button)
            self._mouse.release(resolved_button)
            if index < clicks - 1:
                sleep(resolved_interval_ms / 1000)

    def double_click(self, point: Point, *, button: str = "left") -> None:
        self.click(point, button=button, clicks=2)

    def send_hotkey(self, *keys: str) -> None:
        resolved_keys = [resolve_keyboard_key(key) for key in keys]
        for key in resolved_keys:
            self._keyboard.press(key)
        for key in reversed(resolved_keys):
            self._keyboard.release(key)

    def send_text(self, text: str, *, interval_ms: int = 0) -> None:
        self._set_clipboard_text(text)
        self.send_hotkey("ctrl", "v")
        if interval_ms > 0:
            sleep(interval_ms / 1000)

    def get_window_process_id(self, handle: int) -> int:
        if handle <= 0:
            raise AutomationError("Window handle must be positive.")

        process_id = wintypes.DWORD()
        thread_id = int(self._user32.GetWindowThreadProcessId(handle, ctypes.byref(process_id)))
        if thread_id == 0 or process_id.value <= 0:
            raise AutomationError(f"Failed to read process id for window [{handle}].")
        return int(process_id.value)

    def dismiss_process_popup(self, handle: int, *, timeout_seconds: float = 0.4) -> bool:
        process_id = self.get_window_process_id(handle)
        deadline = monotonic() + timeout_seconds
        dismissed = False

        while monotonic() < deadline:
            button = find_process_popup_button(
                anchor_handle=handle,
                process_id=process_id,
            )
            if button is None:
                if dismissed:
                    return True
                sleep(self._popup_poll_interval_seconds)
                continue
            invoke_process_button(button)
            dismissed = True
            sleep(self._popup_action_delay_seconds)

        return dismissed

    def close_window(self, handle: int) -> None:
        if handle <= 0:
            raise AutomationError("Window handle must be positive.")
        if not self._user32.PostMessageW(handle, WM_CLOSE, 0, 0):
            raise AutomationError(f"Failed to post WM_CLOSE to window [{handle}].")

    def find_windows_by_title(self, title_substring: str) -> list[WindowInfo]:
        normalized = title_substring.strip().casefold()
        if not normalized:
            raise AutomationError("title_substring must be a non-empty string.")

        foreground_handle = int(self._user32.GetForegroundWindow())
        matches: list[WindowInfo] = []
        enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @enum_proc
        def callback(hwnd: wintypes.HWND, _lparam: wintypes.LPARAM) -> bool:
            if not self._user32.IsWindowVisible(hwnd):
                return True
            title = self._read_window_title(hwnd)
            if normalized in title.casefold():
                matches.append(
                    WindowInfo(
                        handle=int(hwnd),
                        title=title,
                        is_foreground=int(hwnd) == foreground_handle,
                    )
                )
            return True

        if not self._user32.EnumWindows(callback, 0):
            raise AutomationError("Failed to enumerate open Windows.")
        return matches

    def wait_for_window(
        self,
        title_substring: str,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float = 0.1,
        known_handles: set[int] | None = None,
    ) -> WindowInfo:
        deadline = monotonic() + timeout_seconds
        known_handles = known_handles or set()
        last_matches: list[WindowInfo] = []

        while True:
            matches = self.find_windows_by_title(title_substring)
            last_matches = matches
            if known_handles:
                fresh_matches = [match for match in matches if match.handle not in known_handles]
                if fresh_matches:
                    return next((match for match in fresh_matches if match.is_foreground), fresh_matches[0])
            elif matches:
                return next((match for match in matches if match.is_foreground), matches[0])

            if monotonic() >= deadline:
                break
            sleep(poll_interval_seconds)

        visible_handles = ", ".join(str(match.handle) for match in last_matches) or "none"
        raise AutomationError(
            f"Timed out waiting for a window containing '{title_substring}'. "
            f"Visible matching handles at timeout: {visible_handles}."
        )

    def window_exists(self, handle: int) -> bool:
        if handle <= 0:
            raise AutomationError("Window handle must be positive.")
        return bool(self._user32.IsWindow(handle))

    def focus_window(self, handle: int) -> None:
        if handle <= 0:
            raise AutomationError("Window handle must be positive.")
        self._user32.ShowWindow(handle, SW_RESTORE)
        if not self._user32.SetForegroundWindow(handle):
            raise AutomationError(f"Failed to focus window [{handle}].")
        sleep(self._focus_delay_seconds)

    def wait_for_window_closed(
        self,
        handle: int,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float = 0.1,
    ) -> None:
        deadline = monotonic() + timeout_seconds
        while self.window_exists(handle):
            if monotonic() >= deadline:
                raise AutomationError(f"Timed out waiting for window [{handle}] to close.")
            sleep(poll_interval_seconds)

    def _set_clipboard_text(self, text: str) -> None:
        result = subprocess.run(
            [
                self._powershell_executable,
                "-NoProfile",
                "-Sta",
                "-Command",
                f"Set-Clipboard -Value @'\n{text}\n'@",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise AutomationError(f"Failed to set clipboard text: {stderr}")

    def _read_window_title(self, hwnd: wintypes.HWND) -> str:
        title_length = int(self._user32.GetWindowTextLengthW(hwnd))
        if title_length == 0:
            return ""
        buffer = ctypes.create_unicode_buffer(title_length + 1)
        copied = self._user32.GetWindowTextW(hwnd, buffer, len(buffer))
        if copied == 0:
            return ""
        return buffer.value


def build_screenshot_script(output_path: Path) -> str:
    quoted_path = _powershell_single_quote(str(output_path))
    return "\n".join(
        [
            'Add-Type @"',
            "using System;",
            "using System.Runtime.InteropServices;",
            "public static class NativeMethods {",
            '    [DllImport("user32.dll")]',
            "    public static extern bool SetProcessDPIAware();",
            "}",
            '"@',
            "[void][NativeMethods]::SetProcessDPIAware()",
            "Add-Type -AssemblyName System.Windows.Forms",
            "Add-Type -AssemblyName System.Drawing",
            "$bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen",
            "$bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height",
            "$graphics = [System.Drawing.Graphics]::FromImage($bitmap)",
            "$graphics.CopyFromScreen($bounds.Left, $bounds.Top, 0, 0, $bitmap.Size)",
            f"$bitmap.Save('{quoted_path}', [System.Drawing.Imaging.ImageFormat]::Png)",
            "$graphics.Dispose()",
            "$bitmap.Dispose()",
        ]
    )


def resolve_keyboard_key(name: str) -> Key | str:
    normalized = name.strip().lower()
    aliases: dict[str, Key | str] = {
        "ctrl": Key.ctrl,
        "control": Key.ctrl,
        "shift": Key.shift,
        "alt": Key.alt,
        "win": Key.cmd,
        "meta": Key.cmd,
        "cmd": Key.cmd,
        "enter": Key.enter,
        "return": Key.enter,
        "tab": Key.tab,
        "esc": Key.esc,
        "escape": Key.esc,
        "space": Key.space,
    }
    if normalized in aliases:
        return aliases[normalized]
    if len(normalized) == 1 and normalized.isprintable():
        return normalized
    if normalized.startswith("f") and normalized[1:].isdigit():
        key = getattr(Key, normalized, None)
        if key is not None:
            return key
    raise AutomationError(f"Unsupported hotkey component: {name}")


def resolve_mouse_button(name: str) -> Button:
    normalized = name.strip().lower()
    if normalized == "left":
        return Button.left
    if normalized == "right":
        return Button.right
    raise AutomationError(f"Unsupported mouse button: {name}")


def _ensure_windows() -> None:
    if sys.platform != "win32" or USER32 is None:
        raise AutomationError("Windows desktop automation is only supported on Windows.")


def _make_process_dpi_aware(user32) -> None:
    set_context = getattr(user32, "SetProcessDpiAwarenessContext", None)
    if set_context is not None and set_context(ctypes.c_void_p(-4)):
        return

    set_aware = getattr(user32, "SetProcessDPIAware", None)
    if set_aware is not None:
        set_aware()


def read_png_dimensions(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    png_signature = b"\x89PNG\r\n\x1a\n"
    if len(raw) < 24 or not raw.startswith(png_signature):
        raise AutomationError(f"Screenshot did not decode as a PNG image: {path}")
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    if width <= 0 or height <= 0:
        raise AutomationError(f"Screenshot reported invalid dimensions: {path}")
    return width, height


def _powershell_single_quote(value: str) -> str:
    return value.replace("'", "''")


def find_process_popup_button(*, anchor_handle: int, process_id: int) -> Any | None:
    desktop = _load_uia_desktop()
    best_match: tuple[int, Any] | None = None

    for root in _iter_process_popup_roots(
        desktop=desktop,
        anchor_handle=anchor_handle,
        process_id=process_id,
    ):
        try:
            candidates = root.descendants()
        except Exception:
            continue

        for button in candidates:
            try:
                if getattr(button.element_info, "control_type", "") != "Button":
                    continue
                label = _popup_button_label(button)
                if label not in POPUP_BUTTON_PRIORITY:
                    continue
                if button.is_visible() and button.is_enabled():
                    popup_ancestor = _find_message_dialog_ancestor(button, root)
                    if popup_ancestor is None:
                        continue
                    score = POPUP_BUTTON_PRIORITY[label]
                    if best_match is None or score < best_match[0]:
                        best_match = (score, button)
            except Exception:
                continue

    return None if best_match is None else best_match[1]


def invoke_process_button(button: Any) -> None:
    try:
        button.invoke()
        return
    except Exception:
        pass

    try:
        button.click_input()
        return
    except Exception as exc:
        raise AutomationError("Failed to invoke a process-owned popup button.") from exc


def _load_uia_desktop():
    try:
        from pywinauto import Desktop
    except ImportError as exc:
        raise AutomationError(
            "The 'pywinauto' package is required for process-based popup handling."
        ) from exc

    return Desktop(backend="uia")


def _iter_process_popup_roots(*, desktop, anchor_handle: int, process_id: int) -> list[Any]:
    roots: list[Any] = []
    seen: set[int] = set()

    def add(candidate: Any) -> None:
        try:
            wrapper = candidate.wrapper_object() if hasattr(candidate, "wrapper_object") else candidate
        except Exception:
            return

        handle = int(getattr(wrapper, "handle", 0) or 0)
        if handle and handle in seen:
            return
        if handle:
            seen.add(handle)
        roots.append(wrapper)

    add(desktop.window(handle=anchor_handle))
    if roots:
        try:
            add(roots[0].top_level_parent())
        except Exception:
            pass

    for window in desktop.windows(process=process_id):
        add(window)

    return roots


def _popup_button_label(button: Any) -> str:
    element_name = getattr(button.element_info, "name", "") or ""
    window_text = button.window_text() or ""
    return (element_name or window_text).strip().casefold()


def _find_message_dialog_ancestor(button: Any, root: Any) -> Any | None:
    root_handle = int(getattr(root, "handle", 0) or 0)
    current = button

    while True:
        try:
            current = current.parent()
        except Exception:
            return None
        if current is None:
            return None

        current_handle = int(getattr(current, "handle", 0) or 0)
        if root_handle and current_handle == root_handle:
            return None
        if _is_message_dialog(current):
            return current


def _is_message_dialog(candidate: Any) -> bool:
    try:
        if getattr(candidate.element_info, "control_type", "") not in {"Window", "Pane"}:
            return False
        descendants = candidate.descendants()
    except Exception:
        return False

    has_message = False
    has_action_button = False
    has_form_controls = False
    for child in descendants:
        try:
            control_type = getattr(child.element_info, "control_type", "")
            text = (getattr(child.element_info, "name", "") or child.window_text() or "").strip().casefold()
        except Exception:
            continue

        if control_type in FORM_CONTROL_TYPES:
            has_form_controls = True
            continue
        if control_type == "Button" and text in POPUP_BUTTON_PRIORITY:
            has_action_button = True
        elif control_type == "Text" and text and text not in POPUP_BUTTON_PRIORITY:
            has_message = True

        if has_form_controls:
            return False
        if has_message and has_action_button:
            return True

    return False
