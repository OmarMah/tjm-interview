from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Callable, Protocol

from tjm_automation.core.errors import GroundingError
from tjm_automation.core.models import TargetSettings, VisionLlmSettings
from tjm_automation.grounding.types import BoundingBox, DetectionResult


ImageSizeReader = Callable[[Path], tuple[int, int]]
ClientFactory = Callable[[VisionLlmSettings, float], "ChatCompletionsClient"]


class ChatCompletionsClient(Protocol):
    def create_completion(self, *, model: str, messages: list[dict[str, Any]]) -> str:
        """Submit a chat completion request and return the assistant message text."""


class VisionLlmGrounder:
    def __init__(
        self,
        *,
        config: VisionLlmSettings,
        min_confidence: float,
        backend_name: str = "vision_llm",
        request_timeout_seconds: float = 60.0,
        client_factory: ClientFactory | None = None,
        image_size_reader: ImageSizeReader | None = None,
    ) -> None:
        self.backend_name = backend_name
        self._config = config
        self._min_confidence = min_confidence
        self._request_timeout_seconds = request_timeout_seconds
        self._client_factory = client_factory or create_openai_chat_completions_client
        self._image_size_reader = image_size_reader or read_png_dimensions

    def locate(self, *, screenshot_path: Path, target: TargetSettings) -> DetectionResult:
        resolved_path = screenshot_path.expanduser().resolve()
        if not resolved_path.exists():
            raise GroundingError(f"Screenshot path does not exist: {resolved_path}")

        image_width, image_height = self._image_size_reader(resolved_path)
        client = self._client_factory(self._config, self._request_timeout_seconds)
        response_text = client.create_completion(
            model=self._config.model,
            messages=build_chat_messages(
                prompt=build_grounding_prompt(
                    target=target,
                    image_width=image_width,
                    image_height=image_height,
                ),
                image_data_url=build_image_data_url(resolved_path),
            ),
        )

        parsed = parse_detection_response_text(response_text)
        if not _coerce_bool(parsed.get("found", False)):
            reason = str(parsed.get("reason", "The model reported that the target was not found."))
            raise GroundingError(reason)

        confidence = float(parsed.get("confidence", 0.0))
        if confidence < self._min_confidence:
            raise GroundingError(
                f"Vision model confidence {confidence:.2f} is below the configured threshold "
                f"{self._min_confidence:.2f}."
            )

        box = build_bounding_box(parsed, image_width=image_width, image_height=image_height)
        matched_text = parsed.get("matched_text")
        if matched_text is not None:
            matched_text = str(matched_text)

        return DetectionResult(
            bounding_box=box,
            center_x=box.center_x,
            center_y=box.center_y,
            confidence=round(confidence, 3),
            backend_name=self.backend_name,
            matched_text=matched_text,
            screenshot_path=resolved_path,
            raw_response_text=response_text,
        )


class OpenAiChatCompletionsClient:
    def __init__(self, config: VisionLlmSettings, timeout_seconds: float) -> None:
        try:
            from openai import APIConnectionError, APIStatusError, OpenAI
        except ImportError as exc:
            raise GroundingError(
                "The 'openai' package is required for the vision_llm backend. "
                "Install project dependencies before running grounding."
            ) from exc

        self._api_connection_error = APIConnectionError
        self._api_status_error = APIStatusError
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=timeout_seconds,
        )

    def create_completion(self, *, model: str, messages: list[dict[str, Any]]) -> str:
        try:
            completion = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
        except self._api_connection_error as exc:
            raise GroundingError(f"Vision LLM connection failed: {exc}") from exc
        except self._api_status_error as exc:
            status_code = getattr(exc, "status_code", "unknown")
            request_id = getattr(exc, "request_id", None)
            suffix = f" request_id={request_id}" if request_id else ""
            raise GroundingError(
                f"Vision LLM request failed with status {status_code}.{suffix}"
            ) from exc

        if not completion.choices:
            raise GroundingError("Vision LLM response did not contain any choices.")

        message = completion.choices[0].message
        return extract_message_text(message.content)


def create_openai_chat_completions_client(
    config: VisionLlmSettings,
    timeout_seconds: float,
) -> ChatCompletionsClient:
    return OpenAiChatCompletionsClient(config, timeout_seconds)


def build_chat_messages(*, prompt: str, image_data_url: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a desktop UI grounding model. Return valid JSON only, with no "
                "markdown fences and no extra explanation."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def build_grounding_prompt(*, target: TargetSettings, image_width: int, image_height: int) -> str:
    return (
        "Find the desktop icon or clickable UI element that best matches the target.\n"
        f"Target display name: {target.display_name}\n"
        f"Grounding query: {target.grounding_query}\n"
        f"Expected window title: {target.window_title_contains}\n"
        f"Image size: width={image_width}, height={image_height}\n"
        "Return a single JSON object with this schema:\n"
        "{"
        '"found": true, '
        '"confidence": 0.0, '
        '"left": 0, '
        '"top": 0, '
        '"width": 0, '
        '"height": 0, '
        '"matched_text": "visible label or null", '
        '"reason": "short reason"'
        "}\n"
        "Use integer coordinates in the original image pixel space.\n"
        "Prefer the clickable icon bounds over the text label bounds when possible.\n"
        "If the target is not visible, return found=false with confidence=0 and a short reason."
    )


def build_image_data_url(path: Path) -> str:
    image_bytes = path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    mime_type = detect_mime_type(path)
    return f"data:{mime_type};base64,{encoded}"


def detect_mime_type(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def parse_detection_response_text(text: str) -> dict[str, Any]:
    json_blob = extract_json_object(text)

    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError as exc:
        raise GroundingError("Vision LLM message did not contain valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise GroundingError("Vision LLM JSON payload was not an object.")
    return parsed


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "type") and getattr(item, "type") == "text":
                parts.append(str(getattr(item, "text", "")))
        if parts:
            return "\n".join(parts).strip()
    raise GroundingError("Vision LLM response did not contain text content.")


def extract_json_object(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise GroundingError("Vision LLM response did not include a JSON object.")
    return candidate[start : end + 1]


def build_bounding_box(
    payload: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> BoundingBox:
    left = max(_coerce_int(payload.get("left")), 0)
    top = max(_coerce_int(payload.get("top")), 0)
    width = max(_coerce_int(payload.get("width")), 1)
    height = max(_coerce_int(payload.get("height")), 1)

    if left >= image_width or top >= image_height:
        raise GroundingError("Vision LLM returned coordinates outside the screenshot bounds.")

    width = min(width, image_width - left)
    height = min(height, image_height - top)
    return BoundingBox(left=left, top=top, width=width, height=height)


def read_png_dimensions(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    png_signature = b"\x89PNG\r\n\x1a\n"
    if len(raw) < 24 or not raw.startswith(png_signature):
        raise GroundingError("Vision LLM backend currently expects PNG screenshots.")
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    if width <= 0 or height <= 0:
        raise GroundingError("Screenshot image reported invalid dimensions.")
    return width, height


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        raise GroundingError("Vision LLM returned an invalid numeric value.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value)
    if isinstance(value, str):
        try:
            return round(float(value.strip()))
        except ValueError as exc:
            raise GroundingError("Vision LLM returned a non-numeric coordinate.") from exc
    raise GroundingError("Vision LLM returned a missing coordinate field.")
