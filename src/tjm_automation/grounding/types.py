from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoundingBox:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center_x(self) -> int:
        return self.left + self.width // 2

    @property
    def center_y(self) -> int:
        return self.top + self.height // 2


@dataclass(frozen=True)
class DetectionResult:
    bounding_box: BoundingBox
    center_x: int
    center_y: int
    confidence: float
    backend_name: str
    matched_text: str | None = None
    screenshot_path: Path | None = None
    debug_image_path: Path | None = None
    raw_response_text: str | None = None
    raw_response_path: Path | None = None
