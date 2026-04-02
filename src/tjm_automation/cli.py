from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tjm_automation.core.errors import (
    AutomationError,
    ConfigurationError,
    GroundingError,
    IntegrationError,
    StorageError,
    TargetError,
)
from tjm_automation.core.logging import configure_logging
from tjm_automation.core.settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tjm-automation",
        description="Run the TJM desktop automation workflow.",
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    settings = _load_cli_settings(args.config_dir, args.target)
    configure_logging(settings.logging.level)

    if args.command == "run-one":
        return _handle_run_one(args, settings)

    if args.command == "run-assignment":
        return _handle_run_assignment(args, settings)

    parser.print_help()
    return 0


def _load_cli_settings(config_dir: Path | None, target: str):
    try:
        return load_settings(config_dir=config_dir, target_name=target)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(2) from exc


def _to_json_ready(value, *, key_name: str | None = None):
    if key_name and key_name.casefold() in {"api_key"}:
        return "<redacted>"
    if isinstance(value, dict):
        return {key: _to_json_ready(item, key_name=key) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _handle_run_one(args, settings) -> int:
    from tjm_automation.automation.runner import AssignmentRunner

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
    from tjm_automation.automation.runner import AssignmentRunner

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
