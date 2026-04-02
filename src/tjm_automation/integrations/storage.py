from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep

from tjm_automation.core.errors import StorageError


@dataclass(frozen=True)
class PreparedOutputPath:
    path: Path
    should_write: bool
    existed: bool
    overwrite_mode: str


def prepare_output_path(
    *,
    base_dir: Path,
    filename: str,
    overwrite_mode: str,
) -> PreparedOutputPath:
    normalized_filename = filename.strip()
    if not normalized_filename:
        raise StorageError("filename must be a non-empty string.")

    resolved_base_dir = base_dir.expanduser().resolve()
    resolved_base_dir.mkdir(parents=True, exist_ok=True)

    candidate = (resolved_base_dir / normalized_filename).resolve()
    if not candidate.is_relative_to(resolved_base_dir):
        raise StorageError("filename must stay within the configured output directory.")

    candidate.parent.mkdir(parents=True, exist_ok=True)
    existed = candidate.exists()

    if existed:
        if candidate.is_dir():
            raise StorageError(f"Output path points to a directory: {candidate}")
        if overwrite_mode == "skip":
            return PreparedOutputPath(
                path=candidate,
                should_write=False,
                existed=True,
                overwrite_mode=overwrite_mode,
            )
        if overwrite_mode == "fail":
            raise StorageError(f"Refusing to overwrite existing file: {candidate}")
        if overwrite_mode == "overwrite":
            candidate.unlink()
        else:
            raise StorageError(f"Unsupported overwrite mode: {overwrite_mode}")

    return PreparedOutputPath(
        path=candidate,
        should_write=True,
        existed=existed,
        overwrite_mode=overwrite_mode,
    )


def wait_for_path(
    path: Path,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 0.2,
) -> Path:
    resolved_path = path.expanduser().resolve()
    deadline = monotonic() + timeout_seconds

    while not resolved_path.exists():
        if monotonic() >= deadline:
            raise StorageError(f"Timed out waiting for file to appear: {resolved_path}")
        sleep(poll_interval_seconds)

    if not resolved_path.is_file():
        raise StorageError(f"Expected a file but found a non-file path: {resolved_path}")
    return resolved_path
