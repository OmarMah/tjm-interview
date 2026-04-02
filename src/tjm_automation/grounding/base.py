from __future__ import annotations

from pathlib import Path
from typing import Protocol

from tjm_automation.core.models import TargetSettings
from tjm_automation.grounding.types import DetectionResult


class Grounder(Protocol):
    backend_name: str

    def locate(self, *, screenshot_path: Path, target: TargetSettings) -> DetectionResult:
        """Locate a configured target inside a screenshot."""
