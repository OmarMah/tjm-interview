from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from tjm_automation.core.errors import GroundingError
from tjm_automation.grounding.types import DetectionResult


def build_debug_output_path(*, artifacts_dir: Path, screenshot_path: Path, target_key: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        artifacts_dir.resolve()
        / "grounding"
        / f"{screenshot_path.stem}_{target_key}_{timestamp}_annotated.png"
    )


def build_response_log_path(debug_image_path: Path) -> Path:
    return debug_image_path.with_name(f"{debug_image_path.stem}_response.txt")


def write_response_log(*, response_text: str, output_path: Path) -> Path:
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(response_text, encoding="utf-8")
    return destination


def annotate_detection(
    *,
    screenshot_path: Path,
    detection: DetectionResult,
    output_path: Path,
    powershell_executable: str = "powershell",
) -> Path:
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    label = detection.matched_text or "match"
    caption = f"{label} | {detection.backend_name} | {detection.confidence:.2f}"
    command = [
        powershell_executable,
        "-NoProfile",
        "-Command",
        _build_annotation_script(
            screenshot_path=screenshot_path.resolve(),
            output_path=destination,
            caption=caption,
            left=detection.bounding_box.left,
            top=detection.bounding_box.top,
            width=detection.bounding_box.width,
            height=detection.bounding_box.height,
            center_x=detection.center_x,
            center_y=detection.center_y,
        ),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise GroundingError(f"Failed to annotate detection image: {stderr}")
    return destination


def _build_annotation_script(
    *,
    screenshot_path: Path,
    output_path: Path,
    caption: str,
    left: int,
    top: int,
    width: int,
    height: int,
    center_x: int,
    center_y: int,
) -> str:
    quoted_input = _powershell_single_quote(str(screenshot_path))
    quoted_output = _powershell_single_quote(str(output_path))
    quoted_caption = _powershell_single_quote(caption)

    return "\n".join(
        [
            "Add-Type -AssemblyName System.Drawing",
            "$source = [System.Drawing.Image]::FromFile('{0}')".format(quoted_input),
            "$bitmap = New-Object System.Drawing.Bitmap $source",
            "$source.Dispose()",
            "$graphics = [System.Drawing.Graphics]::FromImage($bitmap)",
            "$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias",
            "$pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::Lime), 3",
            "$centerPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::Red), 2",
            "$graphics.DrawRectangle($pen, {0}, {1}, {2}, {3})".format(left, top, width, height),
            "$graphics.DrawLine($centerPen, {0}, {1}, {2}, {1})".format(center_x - 8, center_y, center_x + 8),
            "$graphics.DrawLine($centerPen, {0}, {1}, {0}, {2})".format(center_x, center_y - 8, center_y + 8),
            "$font = New-Object System.Drawing.Font('Segoe UI', 12, [System.Drawing.FontStyle]::Bold)",
            "$caption = '{0}'".format(quoted_caption),
            "$measure = $graphics.MeasureString($caption, $font)",
            "$labelWidth = [int][Math]::Ceiling($measure.Width) + 12",
            "$labelHeight = [int][Math]::Ceiling($measure.Height) + 8",
            "$labelTop = [Math]::Max(0, {0} - $labelHeight - 4)".format(top),
            "$labelBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(180, 0, 0, 0))",
            "$graphics.FillRectangle($labelBrush, {0}, $labelTop, $labelWidth, $labelHeight)".format(left),
            "$graphics.DrawString($caption, $font, [System.Drawing.Brushes]::White, [float]({0} + 6), [float]($labelTop + 4))".format(left),
            "$bitmap.Save('{0}', [System.Drawing.Imaging.ImageFormat]::Png)".format(quoted_output),
            "$graphics.Dispose()",
            "$pen.Dispose()",
            "$centerPen.Dispose()",
            "$labelBrush.Dispose()",
            "$font.Dispose()",
            "$bitmap.Dispose()",
        ]
    )


def _powershell_single_quote(value: str) -> str:
    return value.replace("'", "''")
