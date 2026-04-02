"""Target-specific adapters for configurable applications."""

from tjm_automation.core.errors import TargetError
from tjm_automation.core.models import Settings
from tjm_automation.grounding import GroundingPipeline
from tjm_automation.targets.notepad import NotepadTargetAdapter, SavedDocumentResult


def build_target_adapter(
    settings: Settings,
    *,
    desktop_controller=None,
    grounding_pipeline: GroundingPipeline | None = None,
):
    target_key = settings.target.key.strip().casefold()
    if target_key == "notepad":
        return NotepadTargetAdapter(
            settings,
            desktop_controller=desktop_controller,
            grounding_pipeline=grounding_pipeline,
        )
    raise TargetError(f"Unsupported target adapter: {settings.target.key}")


__all__ = ["NotepadTargetAdapter", "SavedDocumentResult", "build_target_adapter"]
