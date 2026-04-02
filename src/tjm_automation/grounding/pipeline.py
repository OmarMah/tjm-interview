from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tjm_automation.core.errors import GroundingError
from tjm_automation.core.models import Settings, TargetSettings
from tjm_automation.grounding.base import Grounder
from tjm_automation.grounding.backends.vision_llm import VisionLlmGrounder
from tjm_automation.grounding.debug import (
    annotate_detection,
    build_debug_output_path,
    build_response_log_path,
    write_response_log,
)
from tjm_automation.grounding.types import DetectionResult


class GroundingPipeline:
    def __init__(self, grounder: Grounder) -> None:
        self._grounder = grounder

    @property
    def backend_name(self) -> str:
        return self._grounder.backend_name

    def locate(
        self,
        *,
        screenshot_path: Path,
        target: TargetSettings,
        artifacts_dir: Path | None = None,
        debug_output_path: Path | None = None,
    ) -> DetectionResult:
        detection = self._grounder.locate(screenshot_path=screenshot_path, target=target)

        resolved_debug_path = debug_output_path
        if resolved_debug_path is None and artifacts_dir is not None:
            resolved_debug_path = build_debug_output_path(
                artifacts_dir=artifacts_dir,
                screenshot_path=screenshot_path,
                target_key=target.key,
            )

        if resolved_debug_path is not None:
            annotate_detection(
                screenshot_path=screenshot_path,
                detection=detection,
                output_path=resolved_debug_path,
            )
            detection = replace(detection, debug_image_path=resolved_debug_path.resolve())

            if detection.raw_response_text:
                response_log_path = write_response_log(
                    response_text=detection.raw_response_text,
                    output_path=build_response_log_path(resolved_debug_path.resolve()),
                )
                detection = replace(detection, raw_response_path=response_log_path)

        return detection


def build_grounding_pipeline(settings: Settings) -> GroundingPipeline:
    return GroundingPipeline(_create_grounder(settings))


def _create_grounder(settings: Settings) -> Grounder:
    backend_name = settings.app.grounding_backend.strip().casefold()
    if backend_name in {"vision_llm", "vlm", "openai_compatible"}:
        if settings.vision_llm is None:
            raise GroundingError("Vision LLM settings were not loaded.")
        return VisionLlmGrounder(
            config=settings.vision_llm,
            min_confidence=settings.app.confidence_threshold,
            backend_name=backend_name,
            request_timeout_seconds=settings.app.grounding_timeout_seconds,
        )
    raise GroundingError(f"Unsupported grounding backend: {settings.app.grounding_backend}")
