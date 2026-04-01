from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tjm_automation import __version__
from tjm_automation.core.errors import (
    AutomationError,
    ConfigurationError,
    GroundingError,
    IntegrationError,
    StorageError,
    TargetError,
)
from tjm_automation.core.logging import configure_logging, get_logger
from tjm_automation.core.settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tjm-automation",
        description="Bootstrap CLI for the TJM desktop automation project.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Optional path to the config directory. Defaults to auto-discovery.",
    )
    parser.add_argument(
        "--target",
        default="notepad",
        help="Target configuration name to load from config/targets.",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("check-config", help="Validate configuration and print a short summary.")
    subparsers.add_parser("show-config", help="Print the resolved configuration as JSON.")
    subparsers.add_parser("version", help="Print the application version.")

    screenshot_parser = subparsers.add_parser(
        "capture-screenshot",
        help="Capture a desktop screenshot into the artifacts directory or a custom path.",
    )
    screenshot_parser.add_argument("--output", type=Path, default=None, help="Optional output path.")
    screenshot_parser.add_argument(
        "--show-desktop",
        action="store_true",
        help="Send Win+D before capturing the screenshot.",
    )

    cursor_parser = subparsers.add_parser(
        "cursor-position",
        help="Print the current mouse cursor coordinates.",
    )
    cursor_parser.add_argument(
        "--show-desktop",
        action="store_true",
        help="Send Win+D before reading the cursor position.",
    )

    window_parser = subparsers.add_parser(
        "check-window",
        help="List visible windows whose title contains the provided text.",
    )
    window_parser.add_argument(
        "--title",
        default=None,
        help="Substring to match. Defaults to the configured target window title.",
    )

    click_parser = subparsers.add_parser(
        "click-point",
        help="Move to and optionally click a screen coordinate.",
    )
    click_parser.add_argument("x", type=int, help="Screen X coordinate.")
    click_parser.add_argument("y", type=int, help="Screen Y coordinate.")
    click_parser.add_argument(
        "--double",
        action="store_true",
        help="Perform a double-click instead of a single click.",
    )
    click_parser.add_argument(
        "--button",
        default="left",
        choices=("left", "right"),
        help="Mouse button to use.",
    )
    click_parser.add_argument(
        "--move-duration-ms",
        type=int,
        default=0,
        help="Optional mouse move duration before clicking.",
    )
    click_parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the click. Without this flag, the command is a dry run.",
    )
    click_parser.add_argument(
        "--show-desktop",
        action="store_true",
        help="Send Win+D before moving the cursor.",
    )

    launch_parser = subparsers.add_parser(
        "launch-target",
        help="Ground the configured target icon, click it, and verify the target window opens.",
    )
    launch_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the launch result as JSON.",
    )

    write_parser = subparsers.add_parser(
        "write-sample",
        help="Launch the configured target, type sample text, save it, and close the app.",
    )
    write_parser.add_argument(
        "--filename",
        default=None,
        help="Optional output filename. Defaults to the configured filename pattern with id=0.",
    )
    content_group = write_parser.add_mutually_exclusive_group()
    content_group.add_argument(
        "--text",
        default=None,
        help="Optional literal text to type instead of the generated sample content.",
    )
    content_group.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="Optional local text file whose contents will be typed into the target app.",
    )
    write_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the save result as JSON.",
    )

    run_one_parser = subparsers.add_parser(
        "run-one",
        help="Fetch one post from the configured API and save it through the target app.",
    )
    run_one_parser.add_argument(
        "--post-id",
        type=int,
        required=True,
        help="Post ID to fetch from the API.",
    )
    run_one_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON.",
    )

    run_assignment_parser = subparsers.add_parser(
        "run-assignment",
        help="Fetch the configured set of posts and save them through the target app.",
    )
    run_assignment_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of posts to process. Defaults to app.post_limit.",
    )
    run_assignment_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the batch result as JSON.",
    )

    detect_parser = subparsers.add_parser(
        "detect-target",
        help="Run the configured grounding backend on a fresh or existing screenshot.",
    )
    detect_parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Optional existing screenshot path. If omitted, a fresh screenshot is captured.",
    )
    detect_parser.add_argument(
        "--show-desktop",
        action="store_true",
        help="Send Win+D before capturing a fresh screenshot.",
    )
    detect_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the detection result as JSON.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    settings = _load_cli_settings(args.config_dir, args.target)
    configure_logging(settings.logging.level)
    logger = get_logger(__name__)

    if args.command in (None, "check-config"):
        logger.info(
            "Configuration loaded for target '%s' using backend '%s'.",
            settings.target.key,
            settings.app.grounding_backend,
        )
        print(
            f"OK: target={settings.target.display_name}, "
            f"backend={settings.app.grounding_backend}, "
            f"config_dir={settings.config_dir}, "
            f"save_dir={settings.output.save_dir}"
        )
        return 0

    if args.command == "show-config":
        print(json.dumps(_to_json_ready(asdict(settings), redact_secrets=True), indent=2))
        return 0

    if args.command == "capture-screenshot":
        return _handle_capture_screenshot(args, settings)

    if args.command == "cursor-position":
        return _handle_cursor_position(args)

    if args.command == "check-window":
        return _handle_check_window(args, settings)

    if args.command == "click-point":
        return _handle_click_point(args)

    if args.command == "launch-target":
        return _handle_launch_target(args, settings)

    if args.command == "write-sample":
        return _handle_write_sample(args, settings)

    if args.command == "run-one":
        return _handle_run_one(args, settings)

    if args.command == "run-assignment":
        return _handle_run_assignment(args, settings)

    if args.command == "detect-target":
        return _handle_detect_target(args, settings)

    parser.print_help()
    return 0


def _load_cli_settings(config_dir: Path | None, target: str):
    try:
        return load_settings(config_dir=config_dir, target_name=target)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(2) from exc


def _to_json_ready(value, *, redact_secrets: bool = False, key_name: str | None = None):
    if redact_secrets and key_name and key_name.casefold() in {"api_key"}:
        return "<redacted>"
    if isinstance(value, dict):
        return {
            key: _to_json_ready(item, redact_secrets=redact_secrets, key_name=key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_to_json_ready(item, redact_secrets=redact_secrets) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _handle_capture_screenshot(args, settings) -> int:
    controller = _create_desktop_controller()
    if args.show_desktop:
        controller.show_desktop()

    output_path = args.output or controller.build_screenshot_path(settings.app.artifacts_dir)
    screenshot = controller.capture_screenshot(output_path)
    print(
        f"Screenshot saved to {screenshot.path} "
        f"({screenshot.width}x{screenshot.height} at offset {screenshot.offset_x},{screenshot.offset_y})"
    )
    return 0


def _handle_cursor_position(args) -> int:
    controller = _create_desktop_controller()
    if args.show_desktop:
        controller.show_desktop()
    point = controller.get_cursor_position()
    print(f"Cursor: x={point.x}, y={point.y}")
    return 0


def _handle_check_window(args, settings) -> int:
    controller = _create_desktop_controller()
    title = args.title or settings.target.window_title_contains
    matches = controller.find_windows_by_title(title)
    print(f"Matches for '{title}': {len(matches)}")
    for match in matches:
        foreground = " foreground" if match.is_foreground else ""
        print(f"- [{match.handle}] {match.title}{foreground}")
    return 0


def _handle_click_point(args) -> int:
    controller = _create_desktop_controller()
    point = _desktop_point(args.x, args.y)

    if not args.execute:
        action = "double-click" if args.double else "single-click"
        print(
            f"Dry run: would move to ({point.x}, {point.y}) and {action} "
            f"with the {args.button} mouse button."
        )
        return 0

    if args.show_desktop:
        controller.show_desktop()

    controller.move_mouse(point, duration_ms=max(args.move_duration_ms, 0))
    if args.double:
        controller.double_click(point, button=args.button)
    else:
        controller.click(point, button=args.button)
    print(f"Executed click at ({point.x}, {point.y}).")
    return 0


def _handle_launch_target(args, settings) -> int:
    from tjm_automation.targets import build_target_adapter

    try:
        launch_result = build_target_adapter(settings).launch_from_desktop()
    except (TargetError, GroundingError, AutomationError) as exc:
        print(f"Launch error: {exc}")
        raise SystemExit(5) from exc

    if args.json:
        print(json.dumps(_to_json_ready(asdict(launch_result)), indent=2))
        return 0

    detection = launch_result.detection
    print(
        f"Launched {settings.target.display_name} with action={launch_result.action}, "
        f"backend={detection.backend_name}, "
        f"confidence={detection.confidence:.3f}, "
        f"center=({detection.center_x}, {detection.center_y})"
    )
    print(f"Window: [{launch_result.opened_window.handle}] {launch_result.opened_window.title}")
    if launch_result.preexisting_window_handles:
        preexisting = ", ".join(map(str, launch_result.preexisting_window_handles))
        print(f"Preexisting handles: {preexisting}")
    if detection.screenshot_path is not None:
        print(f"Screenshot: {detection.screenshot_path}")
    if detection.debug_image_path is not None:
        print(f"Annotated image: {detection.debug_image_path}")
    if detection.raw_response_path is not None:
        print(f"AI response log: {detection.raw_response_path}")
    return 0


def _handle_write_sample(args, settings) -> int:
    from tjm_automation.targets import build_target_adapter

    try:
        content = _resolve_sample_text(args, settings)
        filename = args.filename or settings.target.filename_pattern.format(id=0)
        save_result = build_target_adapter(settings).write_text_file(
            content=content,
            filename=filename,
        )
    except OSError as exc:
        print(f"Write error: failed to read local text content: {exc}")
        raise SystemExit(6) from exc
    except (TargetError, GroundingError, AutomationError, StorageError) as exc:
        print(f"Write error: {exc}")
        raise SystemExit(6) from exc

    if args.json:
        print(json.dumps(_to_json_ready(asdict(save_result)), indent=2))
        return 0

    if save_result.status == "skipped":
        print(f"Skipped existing file: {save_result.path}")
        return 0

    print(f"Saved sample document to {save_result.path}")
    if save_result.launch_result is not None:
        detection = save_result.launch_result.detection
        print(
            f"Launched with backend={detection.backend_name}, "
            f"confidence={detection.confidence:.3f}, "
            f"center=({detection.center_x}, {detection.center_y})"
        )
        if detection.debug_image_path is not None:
            print(f"Annotated image: {detection.debug_image_path}")
        if detection.raw_response_path is not None:
            print(f"AI response log: {detection.raw_response_path}")
    return 0


def _handle_detect_target(args, settings) -> int:
    from tjm_automation.grounding import build_grounding_pipeline

    screenshot_path = args.screenshot
    if screenshot_path is None:
        controller = _create_desktop_controller()
        if args.show_desktop:
            controller.show_desktop()
        screenshot = controller.capture_screenshot(
            controller.build_screenshot_path(
                settings.app.artifacts_dir,
                prefix=f"{settings.target.key}_detect",
            )
        )
        screenshot_path = screenshot.path
    else:
        screenshot_path = screenshot_path.expanduser().resolve()

    try:
        detection = build_grounding_pipeline(settings).locate(
            screenshot_path=screenshot_path,
            target=settings.target,
            artifacts_dir=settings.app.artifacts_dir,
        )
    except GroundingError as exc:
        print(f"Grounding error: {exc}")
        raise SystemExit(4) from exc

    if args.json:
        print(json.dumps(_to_json_ready(asdict(detection)), indent=2))
        return 0

    print(
        f"Detected {settings.target.display_name} with backend={detection.backend_name}, "
        f"confidence={detection.confidence:.3f}, "
        f"center=({detection.center_x}, {detection.center_y}), "
        f"box=({detection.bounding_box.left}, {detection.bounding_box.top}, "
        f"{detection.bounding_box.width}, {detection.bounding_box.height})"
    )
    print(f"Screenshot: {detection.screenshot_path}")
    if detection.debug_image_path is not None:
        print(f"Annotated image: {detection.debug_image_path}")
    if detection.raw_response_path is not None:
        print(f"AI response log: {detection.raw_response_path}")
    if detection.matched_text:
        print(f"Matched text: {detection.matched_text}")
    return 0


def _handle_run_one(args, settings) -> int:
    from tjm_automation.automation import AssignmentRunner

    try:
        result = AssignmentRunner(settings).run_one(post_id=args.post_id)
    except (
        AutomationError,
        ConfigurationError,
        GroundingError,
        IntegrationError,
        StorageError,
        TargetError,
    ) as exc:
        print(f"Run error: {exc}")
        raise SystemExit(7) from exc

    if args.json:
        print(json.dumps(_to_json_ready(asdict(result)), indent=2))
        return 0

    print(f"Processed post {result.post_id}: {result.status} -> {result.path}")
    print(f"Filename: {result.filename}")
    print(f"Attempts: {result.attempts}")
    print(f"Title: {result.title}")
    return 0


def _handle_run_assignment(args, settings) -> int:
    from tjm_automation.automation import AssignmentRunner

    try:
        result = AssignmentRunner(settings).run_assignment(limit=args.limit)
    except (
        AutomationError,
        ConfigurationError,
        GroundingError,
        IntegrationError,
        StorageError,
        TargetError,
    ) as exc:
        print(f"Run error: {exc}")
        raise SystemExit(7) from exc

    if args.json:
        print(json.dumps(_to_json_ready(asdict(result)), indent=2))
        return 0

    print(
        f"Processed {result.total_posts} posts: "
        f"saved={result.saved_count}, skipped={result.skipped_count}"
    )
    for item in result.results:
        print(f"- post {item.post_id}: {item.status} -> {item.path}")
    return 0


def _create_desktop_controller():
    from tjm_automation.automation.desktop import WindowsDesktopController

    try:
        return WindowsDesktopController()
    except AutomationError as exc:
        print(f"Automation error: {exc}")
        raise SystemExit(3) from exc


def _desktop_point(x: int, y: int):
    from tjm_automation.automation.desktop import Point

    return Point(x=x, y=y)


def _resolve_sample_text(args, settings) -> str:
    if args.text_file is not None:
        return args.text_file.expanduser().resolve().read_text(encoding="utf-8")
    if args.text is not None:
        return args.text
    return settings.target.content_template.format(
        id=0,
        title="Sample Title",
        body="Sample body generated by tjm_automation.",
    )
