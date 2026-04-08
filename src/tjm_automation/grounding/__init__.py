"""Grounding interfaces and pipeline utilities."""

from tjm_automation.grounding.pipeline import GroundingPipeline, build_grounding_pipeline
from tjm_automation.grounding.types import BoundingBox, DetectionResult

__all__ = [
    "BoundingBox",
    "DetectionResult",
    "GroundingPipeline",
    "build_grounding_pipeline",
]
