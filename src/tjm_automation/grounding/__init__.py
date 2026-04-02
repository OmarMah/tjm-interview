"""Grounding interfaces and pipeline utilities."""

from tjm_automation.grounding.pipeline import GroundingPipeline, build_grounding_pipeline
from tjm_automation.grounding.types import BoundingBox, DetectionResult, TextRegion

__all__ = [
    "BoundingBox",
    "DetectionResult",
    "GroundingPipeline",
    "TextRegion",
    "build_grounding_pipeline",
]
