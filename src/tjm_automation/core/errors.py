class TjmAutomationError(Exception):
    """Base exception for the project."""


class ConfigurationError(TjmAutomationError):
    """Raised when configuration cannot be loaded or validated."""


class AutomationError(TjmAutomationError):
    """Raised when a Windows automation action fails."""


class GroundingError(TjmAutomationError):
    """Raised when target grounding fails or cannot be completed."""


class TargetError(TjmAutomationError):
    """Raised when a target-specific workflow step fails."""


class StorageError(TjmAutomationError):
    """Raised when output file preparation or persistence fails."""
