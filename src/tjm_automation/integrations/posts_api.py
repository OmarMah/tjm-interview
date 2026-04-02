from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from shutil import which
from urllib.parse import urlencode

from tjm_automation.core.errors import IntegrationError


@dataclass(frozen=True)
class PostRecord:
    id: int
    user_id: int
    title: str
    body: str


class JsonPlaceholderPostsClient:
    """Fetches posts from the configured JSONPlaceholder-compatible API."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 15.0,
        browser_path: str | None = None,
    ) -> None:
        normalized_base_url = base_url.strip().rstrip("/")
        if not normalized_base_url:
            raise IntegrationError("API base URL must be a non-empty string.")

        self._base_url = normalized_base_url
        self._timeout_seconds = timeout_seconds
        self._browser_path = browser_path or _find_browser()

    def fetch_posts(self, *, limit: int) -> list[PostRecord]:
        if limit < 1:
            raise IntegrationError("limit must be at least 1.")

        query = urlencode(
            {
                "_sort": "id",
                "_order": "asc",
                "_limit": limit,
            }
        )
        payload = self._read_json_via_browser(f"{self._base_url}/posts?{query}")
        if not isinstance(payload, list):
            raise IntegrationError("Expected the posts API to return a JSON array.")

        return [self._parse_post(item, context="posts list") for item in payload]

    def fetch_post(self, *, post_id: int) -> PostRecord:
        if post_id < 1:
            raise IntegrationError("post_id must be at least 1.")

        payload = self._read_json_via_browser(f"{self._base_url}/posts/{post_id}")
        if not isinstance(payload, dict):
            raise IntegrationError("Expected the post API to return a JSON object.")
        return self._parse_post(payload, context=f"post {post_id}")

    def _read_json_via_browser(self, url: str):
        if self._browser_path is None:
            raise IntegrationError("Could not find Chrome or Edge for browser-backed API fetches.")

        try:
            result = subprocess.run(
                [
                    self._browser_path,
                    "--headless=new",
                    "--disable-gpu",
                    "--dump-dom",
                    url,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self._timeout_seconds,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise IntegrationError(f"Browser request timed out for {url}") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise IntegrationError(f"Browser request failed for {url}: {exc}") from exc

        try:
            return _parse_browser_json(result.stdout)
        except json.JSONDecodeError as exc:
            response_preview = result.stdout.strip()
            if len(response_preview) > 1200:
                response_preview = f"{response_preview[:1200]}\n... [truncated]"
            raise IntegrationError(
                f"Browser response was not valid JSON: {url}\n\n{response_preview}"
            ) from exc

    def _parse_post(self, payload: object, *, context: str) -> PostRecord:
        if not isinstance(payload, dict):
            raise IntegrationError(f"Expected {context} items to be JSON objects.")

        post_id = payload.get("id")
        user_id = payload.get("userId")
        title = payload.get("title")
        body = payload.get("body")

        if isinstance(post_id, bool) or not isinstance(post_id, int):
            raise IntegrationError(f"Expected an integer 'id' in {context}.")
        if isinstance(user_id, bool) or not isinstance(user_id, int):
            raise IntegrationError(f"Expected an integer 'userId' in {context}.")
        if not isinstance(title, str):
            raise IntegrationError(f"Expected a string 'title' in {context}.")
        if not isinstance(body, str):
            raise IntegrationError(f"Expected a string 'body' in {context}.")

        return PostRecord(
            id=post_id,
            user_id=user_id,
            title=title,
            body=body,
        )


def _find_browser() -> str | None:
    candidates = [
        which("chrome"),
        which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    return None


def _parse_browser_json(raw_output: str):
    candidates: list[str] = []

    def add_candidate(value: str) -> None:
        cleaned = value.strip().lstrip("\ufeff").strip()
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    add_candidate(raw_output)

    pre_match = re.search(r"<pre[^>]*>(.*?)</pre>", raw_output, flags=re.IGNORECASE | re.DOTALL)
    if pre_match:
        add_candidate(unescape(pre_match.group(1)))

    for opening, closing in (("[", "]"), ("{", "}")):
        start = raw_output.find(opening)
        end = raw_output.rfind(closing)
        if start != -1 and end != -1 and end > start:
            add_candidate(unescape(raw_output[start : end + 1]))

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON payload found in browser response.", raw_output, 0)


__all__ = ["JsonPlaceholderPostsClient", "PostRecord"]
