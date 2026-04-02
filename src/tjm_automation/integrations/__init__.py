"""External integrations for APIs, storage, and diagnostics."""

from tjm_automation.integrations.posts_api import JsonPlaceholderPostsClient, PostRecord
from tjm_automation.integrations.storage import PreparedOutputPath, prepare_output_path, wait_for_path

__all__ = [
    "JsonPlaceholderPostsClient",
    "PostRecord",
    "PreparedOutputPath",
    "prepare_output_path",
    "wait_for_path",
]
