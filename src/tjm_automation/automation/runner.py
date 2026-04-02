from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import sleep

from tjm_automation.core.errors import (
    AutomationError,
    ConfigurationError,
    GroundingError,
    StorageError,
    TargetError,
)
from tjm_automation.core.models import Settings
from tjm_automation.integrations.posts_api import JsonPlaceholderPostsClient, PostRecord
from tjm_automation.targets import SavedDocumentResult, build_target_adapter


@dataclass(frozen=True)
class RenderedPost:
    post: PostRecord
    filename: str
    content: str


@dataclass(frozen=True)
class ProcessedPostResult:
    post_id: int
    user_id: int
    title: str
    filename: str
    path: Path
    status: str
    attempts: int


@dataclass(frozen=True)
class AssignmentRunResult:
    total_posts: int
    saved_count: int
    skipped_count: int
    results: tuple[ProcessedPostResult, ...]


class AssignmentRunner:
    """Coordinates API fetching and target-driven file creation."""

    def __init__(
        self,
        settings: Settings,
        *,
        posts_client: JsonPlaceholderPostsClient | None = None,
        target_adapter=None,
    ) -> None:
        self._settings = settings
        self._posts = posts_client or JsonPlaceholderPostsClient(
            base_url=settings.app.api_base_url,
            timeout_seconds=settings.app.api_timeout_seconds,
        )
        self._target = target_adapter or build_target_adapter(settings)

    def run_one(self, *, post_id: int) -> ProcessedPostResult:
        post = self._posts.fetch_post(post_id=post_id)
        rendered = self._render_post(post)
        saved, attempts = self._write_with_retries(rendered)
        return self._to_processed_result(rendered, saved, attempts=attempts)

    def run_assignment(self, *, limit: int | None = None) -> AssignmentRunResult:
        resolved_limit = self._settings.app.post_limit if limit is None else limit
        posts = self._posts.fetch_posts(limit=resolved_limit)
        results = tuple(self._run_rendered_post(self._render_post(post)) for post in posts)
        saved_count = sum(1 for result in results if result.status == "saved")
        skipped_count = sum(1 for result in results if result.status == "skipped")
        return AssignmentRunResult(
            total_posts=len(results),
            saved_count=saved_count,
            skipped_count=skipped_count,
            results=results,
        )

    def _run_rendered_post(self, rendered: RenderedPost) -> ProcessedPostResult:
        saved, attempts = self._write_with_retries(rendered)
        return self._to_processed_result(rendered, saved, attempts=attempts)

    def _render_post(self, post: PostRecord) -> RenderedPost:
        context = {
            "id": post.id,
            "user_id": post.user_id,
            "title": post.title,
            "body": post.body,
        }
        return RenderedPost(
            post=post,
            filename=self._format_template(
                self._settings.target.filename_pattern,
                context=context,
                source="target.filename_pattern",
            ),
            content=self._format_template(
                self._settings.target.content_template,
                context=context,
                source="target.content_template",
            ),
        )

    def _write_with_retries(self, rendered: RenderedPost) -> tuple[SavedDocumentResult, int]:
        last_error = None
        max_attempts = self._settings.app.retry_attempts

        for attempt in range(1, max_attempts + 1):
            try:
                saved = self._target.write_text_file(
                    content=rendered.content,
                    filename=rendered.filename,
                )
                return saved, attempt
            except (AutomationError, GroundingError, StorageError, TargetError) as exc:
                last_error = exc
                if attempt >= max_attempts:
                    raise
                if self._settings.app.retry_delay_seconds > 0:
                    sleep(self._settings.app.retry_delay_seconds)

        assert last_error is not None
        raise last_error

    def _to_processed_result(
        self,
        rendered: RenderedPost,
        saved: SavedDocumentResult,
        *,
        attempts: int,
    ) -> ProcessedPostResult:
        return ProcessedPostResult(
            post_id=rendered.post.id,
            user_id=rendered.post.user_id,
            title=rendered.post.title,
            filename=rendered.filename,
            path=saved.path,
            status=saved.status,
            attempts=attempts,
        )

    def _format_template(
        self,
        template: str,
        *,
        context: dict[str, object],
        source: str,
    ) -> str:
        try:
            return template.format(**context)
        except (KeyError, ValueError) as exc:
            raise ConfigurationError(f"Invalid placeholder usage in {source}: {exc}") from exc


__all__ = ["AssignmentRunResult", "AssignmentRunner", "ProcessedPostResult", "RenderedPost"]
