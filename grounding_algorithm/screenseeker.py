from __future__ import annotations

import argparse
import base64
import difflib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any

import cv2
import numpy as np
from openai import OpenAI


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_SCREEN_WIDTH = 1920
DEFAULT_SCREEN_HEIGHT = 1080
REQUEST_IMAGE_MAX_SIDE = 1600
LEAF_IMAGE_MAX_SIDE = 1280
DILATION_FACTOR = 1.6
MIN_REGION_SIZE = 256
MAX_DEPTH = 3
PLANNER_CANDIDATES = 5
NMS_THRESHOLD = 0.45
MIN_TEXT_EVIDENCE = 0.25
PLANNER_SUPPORT_WEIGHT = 0.75
MIN_ICON_SIDE = 18
DIRECT_CV_REFINEMENT_MAX_SIDE = 256
SEGMENT_SEARCH_X_FACTOR = 0.35
SEGMENT_SEARCH_TOP_FACTOR = 0.2
SEGMENT_SEARCH_BOTTOM_FACTOR = 0.55
EDGE_LOW_THRESHOLD = 40
EDGE_HIGH_THRESHOLD = 120
BACKGROUND_DISTANCE_THRESHOLD = 18.0
MIN_SEGMENT_AREA = 64
MIN_FOREGROUND_PIXELS_PER_LINE = 3
TRIM_PADDING = 1


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center_x(self) -> float:
        return self.left + self.width / 2

    @property
    def center_y(self) -> float:
        return self.top + self.height / 2


@dataclass(frozen=True)
class Candidate:
    box: Box
    confidence: float
    score: float
    matched_text: str | None = None
    text_score: float = 0.0


@dataclass(frozen=True)
class SearchResult:
    target: str
    screenshot_path: Path
    annotated_path: Path
    box: Box
    confidence: float
    depth: int
    matched_text: str | None = None
    score: float = 0.0
    planner_support: float = 0.0
    leaf_text_score: float = 0.0


class ScreenSeeker:
    def __init__(
        self,
        *,
        planner_base_url: str,
        planner_api_key: str,
        planner_model: str,
        grounding_base_url: str,
        grounding_api_key: str,
        grounding_model: str,
        output_dir: Path,
        powershell_executable: str = "powershell",
        screen_width: int = DEFAULT_SCREEN_WIDTH,
        screen_height: int = DEFAULT_SCREEN_HEIGHT,
        max_depth: int = MAX_DEPTH,
        leaf_size: int = LEAF_IMAGE_MAX_SIDE,
        request_image_max_side: int = REQUEST_IMAGE_MAX_SIDE,
        dilation_factor: float = DILATION_FACTOR,
        min_region_size: int = MIN_REGION_SIZE,
        planner_candidates: int = PLANNER_CANDIDATES,
        nms_threshold: float = NMS_THRESHOLD,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._planner_client = OpenAI(
            api_key=planner_api_key,
            base_url=normalize_base_url(planner_base_url),
            timeout=timeout_seconds,
        )
        self._grounding_client = OpenAI(
            api_key=grounding_api_key,
            base_url=normalize_base_url(grounding_base_url),
            timeout=timeout_seconds,
        )
        self._planner_model = planner_model
        self._grounding_model = grounding_model
        self._output_dir = output_dir.expanduser().resolve()
        self._powershell_executable = powershell_executable
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._max_depth = max_depth
        self._leaf_size = leaf_size
        self._request_image_max_side = request_image_max_side
        self._dilation_factor = dilation_factor
        self._min_region_size = min_region_size
        self._planner_candidates = planner_candidates
        self._nms_threshold = nms_threshold

    def locate(self, *, target_name: str) -> SearchResult:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = self._output_dir / f"{target_name.replace(' ', '_').casefold()}_{stamp}"
        work_dir.mkdir(parents=True, exist_ok=True)
        resolved_screenshot = work_dir / "desktop.png"
        show_desktop(self._powershell_executable)
        capture_desktop_screenshot(
            output_path=resolved_screenshot,
            powershell_executable=self._powershell_executable,
        )
        width, height = read_png_dimensions(resolved_screenshot)
        if (width, height) != (self._screen_width, self._screen_height):
            raise RuntimeError(
                f"Expected a {self._screen_width}x{self._screen_height} desktop screenshot, "
                f"but captured {width}x{height}."
            )

        root_box = Box(left=0, top=0, width=width, height=height)
        result = self._search(
            patch_path=resolved_screenshot,
            patch_box=root_box,
            target_name=target_name,
            depth=0,
            work_dir=work_dir,
            path_support=0.0,
        )
        if result is None:
            raise RuntimeError(f"Could not locate '{target_name}' in {resolved_screenshot}")

        annotated_path = work_dir / f"{resolved_screenshot.stem}_{target_name.replace(' ', '_')}_annotated.png"
        annotate_box(
            screenshot_path=resolved_screenshot,
            box=result.box,
            caption=f"{target_name} | screenseeker | {result.confidence:.2f}",
            output_path=annotated_path,
            powershell_executable=self._powershell_executable,
        )
        return SearchResult(
            target=target_name,
            screenshot_path=resolved_screenshot,
            annotated_path=annotated_path,
            box=result.box,
            confidence=result.confidence,
            depth=result.depth,
            matched_text=result.matched_text,
            score=result.score,
        )

    def _search(
        self,
        *,
        patch_path: Path,
        patch_box: Box,
        target_name: str,
        depth: int,
        work_dir: Path,
        path_support: float,
    ) -> SearchResult | None:
        patch_width, patch_height = read_png_dimensions(patch_path)
        if max(patch_width, patch_height) <= self._leaf_size or depth >= self._max_depth:
            return self._ground_leaf(
                patch_path=patch_path,
                patch_box=patch_box,
                target_name=target_name,
                depth=depth,
                work_dir=work_dir,
                path_support=path_support,
            )

        candidates = self._plan_candidates(
            patch_path=patch_path,
            patch_width=patch_width,
            patch_height=patch_height,
            target_name=target_name,
            work_dir=work_dir,
            depth=depth,
        )
        if not candidates:
            return self._ground_leaf(
                patch_path=patch_path,
                patch_box=patch_box,
                target_name=target_name,
                depth=depth,
                work_dir=work_dir,
                path_support=path_support,
            )

        results: list[SearchResult] = []
        for index, candidate in enumerate(candidates):
            crop_path = work_dir / f"depth_{depth}_candidate_{index}.png"
            crop_image(
                source_path=patch_path,
                crop_box=candidate.box,
                output_path=crop_path,
                powershell_executable=self._powershell_executable,
            )
            next_box = Box(
                left=patch_box.left + candidate.box.left,
                top=patch_box.top + candidate.box.top,
                width=candidate.box.width,
                height=candidate.box.height,
            )
            result = self._search(
                patch_path=crop_path,
                patch_box=next_box,
                target_name=target_name,
                depth=depth + 1,
                work_dir=work_dir,
                path_support=path_support + candidate.score,
            )
            if result is not None:
                results.append(result)

        if results:
            return choose_best_result(results)

        return self._ground_leaf(
            patch_path=patch_path,
            patch_box=patch_box,
            target_name=target_name,
            depth=depth,
            work_dir=work_dir,
            path_support=path_support,
        )

    def _plan_candidates(
        self,
        *,
        patch_path: Path,
        patch_width: int,
        patch_height: int,
        target_name: str,
        work_dir: Path,
        depth: int,
    ) -> list[Candidate]:
        request_path, scale = prepare_request_image(
            source_path=patch_path,
            output_path=work_dir / f"depth_{depth}_planner_request.png",
            max_side=self._request_image_max_side,
            powershell_executable=self._powershell_executable,
        )
        request_width, request_height = read_png_dimensions(request_path)
        prompt = (
            "You are ScreenSeekeR planner.\n"
            f"Target desktop app icon: {target_name}\n"
            f"Desktop resolution: width={self._screen_width}, height={self._screen_height}\n"
            f"Image size: width={request_width}, height={request_height}\n"
            f"Return up to {self._planner_candidates} likely regions that may contain that desktop icon.\n"
            "Prefer regions that contain both the icon and its visible label text.\n"
            "These should be broader search regions, not tiny exact boxes.\n"
            "Return valid JSON only in this format:\n"
            '{"candidates":[{"left":0,"top":0,"width":0,"height":0,"confidence":0.0,"matched_text":"visible label or null"}]}\n'
            "Use integer coordinates in this image's pixel space."
        )
        payload = self._call_json(
            client=self._planner_client,
            model=self._planner_model,
            prompt=prompt,
            image_path=request_path,
        )
        raw_candidates = payload.get("candidates", [])
        if not isinstance(raw_candidates, list):
            return []

        planned: list[Candidate] = []
        for item in raw_candidates[: self._planner_candidates]:
            if not isinstance(item, dict):
                continue
            box = scale_box(
                Box(
                    left=max(int(item.get("left", 0)), 0),
                    top=max(int(item.get("top", 0)), 0),
                    width=max(int(item.get("width", 1)), 1),
                    height=max(int(item.get("height", 1)), 1),
                ),
                inverse_scale=scale,
                max_width=patch_width,
                max_height=patch_height,
            )
            dilated = dilate_box(
                box=box,
                image_width=patch_width,
                image_height=patch_height,
                factor=self._dilation_factor,
                min_size=self._min_region_size,
            )
            confidence = float(item.get("confidence", 0.5))
            matched_text = read_matched_text(item.get("matched_text"))
            text_score = text_match_score(target_name, matched_text)
            score = score_candidate(
                box=dilated,
                image_width=patch_width,
                image_height=patch_height,
                confidence=confidence,
                matched_text=matched_text,
                target_name=target_name,
            )
            planned.append(
                Candidate(
                    box=dilated,
                    confidence=confidence,
                    score=score,
                    matched_text=matched_text,
                    text_score=text_score,
                )
            )

        ranked = sorted(planned, key=lambda candidate: candidate.score, reverse=True)
        return apply_nms(ranked, threshold=self._nms_threshold)

    def _ground_leaf(
        self,
        *,
        patch_path: Path,
        patch_box: Box,
        target_name: str,
        depth: int,
        work_dir: Path,
        path_support: float,
    ) -> SearchResult | None:
        request_path, scale = prepare_request_image(
            source_path=patch_path,
            output_path=work_dir / f"depth_{depth}_leaf_request.png",
            max_side=self._leaf_size,
            powershell_executable=self._powershell_executable,
        )
        request_width, request_height = read_png_dimensions(request_path)
        prompt = (
            "You are ScreenSeekeR grounding step.\n"
            f"Target desktop app icon: {target_name}\n"
            f"Desktop resolution: width={self._screen_width}, height={self._screen_height}\n"
            f"Image size: width={request_width}, height={request_height}\n"
            "Return valid JSON only in this format:\n"
            '{"found":true,"confidence":0.0,"icon_box":{"left":0,"top":0,"width":0,"height":0}}\n'
            "Return found=true only if the requested desktop icon graphic is visible in this patch.\n"
            "The icon_box must tightly bound only the icon graphic.\n"
            "Do not include the label text, surrounding whitespace, or background.\n"
            "If the icon is not visible, return found=false with zero confidence."
        )
        payload = self._call_json(
            client=self._grounding_client,
            model=self._grounding_model,
            prompt=prompt,
            image_path=request_path,
        )
        if not payload.get("found", False):
            return None

        relative_icon_box = read_box(payload.get("icon_box"))
        if relative_icon_box is None:
            return None
        relative_icon_box = scale_box(
            relative_icon_box,
            inverse_scale=scale,
            max_width=patch_box.width,
            max_height=patch_box.height,
        )
        relative_icon_box = refine_icon_box_with_cv(
            image_path=patch_path,
            coarse_box=relative_icon_box,
            output_prefix=work_dir / f"{patch_path.stem}_cv",
        )
        absolute_box = offset_box(relative_icon_box, left=patch_box.left, top=patch_box.top)
        confidence = float(payload.get("confidence", 0.0))
        matched_text = None
        leaf_text_score = 0.0
        if not icon_box_has_signal(absolute_box):
            return None
        score = score_leaf_result(
            box=absolute_box,
            image_width=patch_box.width,
            image_height=patch_box.height,
            confidence=confidence,
            matched_text=matched_text,
            target_name=target_name,
        )
        score = score + (path_support * PLANNER_SUPPORT_WEIGHT)

        return SearchResult(
            target=target_name,
            screenshot_path=patch_path,
            annotated_path=work_dir / "unused.png",
            box=absolute_box,
            confidence=confidence,
            depth=depth,
            matched_text=matched_text,
            score=score,
            planner_support=path_support,
            leaf_text_score=leaf_text_score,
        )

    def _call_json(self, *, client: OpenAI, model: str, prompt: str, image_path: Path) -> dict[str, Any]:
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Return valid JSON only. No markdown. No explanation.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": build_image_data_url(image_path)}},
                    ],
                },
            ],
        )
        content = completion.choices[0].message.content
        if isinstance(content, str):
            return json.loads(extract_json_object(content))

        text_parts: list[str] = []
        for item in content:
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text_parts.append(getattr(item, "text", ""))
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return json.loads(extract_json_object("\n".join(text_parts)))


def build_image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def extract_json_object(text: str) -> str:
    candidate = text.strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    return candidate[start : end + 1]


def read_box(value: Any) -> Box | None:
    if not isinstance(value, dict):
        return None
    left = max(int(value.get("left", 0)), 0)
    top = max(int(value.get("top", 0)), 0)
    width = max(int(value.get("width", 1)), 1)
    height = max(int(value.get("height", 1)), 1)
    return Box(left=left, top=top, width=width, height=height)


def scale_box(box: Box, *, inverse_scale: float, max_width: int, max_height: int) -> Box:
    left = max(int(round(box.left / inverse_scale)), 0)
    top = max(int(round(box.top / inverse_scale)), 0)
    width = max(int(round(box.width / inverse_scale)), 1)
    height = max(int(round(box.height / inverse_scale)), 1)
    width = min(width, max_width - left)
    height = min(height, max_height - top)
    return Box(left=left, top=top, width=width, height=height)


def dilate_box(
    *,
    box: Box,
    image_width: int,
    image_height: int,
    factor: float,
    min_size: int,
) -> Box:
    center_x = box.center_x
    center_y = box.center_y
    width = min(max(int(round(box.width * factor)), min_size), image_width)
    height = min(max(int(round(box.height * factor)), min_size), image_height)
    left = max(int(round(center_x - width / 2)), 0)
    top = max(int(round(center_y - height / 2)), 0)
    if left + width > image_width:
        left = image_width - width
    if top + height > image_height:
        top = image_height - height
    return Box(left=left, top=top, width=width, height=height)


def offset_box(box: Box, *, left: int, top: int) -> Box:
    return Box(left=box.left + left, top=box.top + top, width=box.width, height=box.height)


def expand_search_box(box: Box, *, image_width: int, image_height: int) -> Box:
    pad_x = max(int(round(box.width * SEGMENT_SEARCH_X_FACTOR)), 12)
    pad_top = max(int(round(box.height * SEGMENT_SEARCH_TOP_FACTOR)), 8)
    pad_bottom = max(int(round(box.height * SEGMENT_SEARCH_BOTTOM_FACTOR)), 20)
    left = max(box.left - pad_x, 0)
    top = max(box.top - pad_top, 0)
    right = min(box.right + pad_x, image_width)
    bottom = min(box.bottom + pad_bottom, image_height)
    return Box(left=left, top=top, width=right - left, height=bottom - top)


def refine_icon_box_with_cv(*, image_path: Path, coarse_box: Box, output_prefix: Path) -> Box:
    image = cv2.imread(str(image_path))
    if image is None:
        return coarse_box

    image_height, image_width = image.shape[:2]
    if max(image_width, image_height) <= DIRECT_CV_REFINEMENT_MAX_SIDE:
        search_box = Box(left=0, top=0, width=image_width, height=image_height)
    else:
        search_box = expand_search_box(coarse_box, image_width=image_width, image_height=image_height)
    roi = image[search_box.top : search_box.bottom, search_box.left : search_box.right]
    if roi.size == 0:
        return coarse_box

    coarse_relative = Box(
        left=coarse_box.left - search_box.left,
        top=coarse_box.top - search_box.top,
        width=coarse_box.width,
        height=coarse_box.height,
    )

    edges = build_edge_map(roi)
    mask = build_foreground_mask(roi=roi, edges=edges)
    component_box = find_best_component(mask=mask, roi=roi, edges=edges, coarse_box=coarse_relative)
    if component_box is None:
        save_cv_debug_images(
            roi=roi,
            mask=mask,
            coarse_box=coarse_relative,
            refined_box=None,
            output_prefix=output_prefix,
        )
        return coarse_box

    refined_component = trim_component_mask(mask=mask, component_box=component_box)
    save_cv_debug_images(
        roi=roi,
        mask=mask,
        coarse_box=coarse_relative,
        refined_box=refined_component,
        output_prefix=output_prefix,
    )
    return offset_box(refined_component, left=search_box.left, top=search_box.top)


def build_edge_map(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, EDGE_LOW_THRESHOLD, EDGE_HIGH_THRESHOLD)
    return cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)


def build_foreground_mask(*, roi: np.ndarray, edges: np.ndarray) -> np.ndarray:
    border_pixels = np.concatenate(
        [roi[0, :, :], roi[-1, :, :], roi[:, 0, :], roi[:, -1, :]],
        axis=0,
    )
    background = np.median(border_pixels.astype(np.float32), axis=0)
    diff = np.linalg.norm(roi.astype(np.float32) - background, axis=2)
    color_mask = (diff > BACKGROUND_DISTANCE_THRESHOLD).astype(np.uint8) * 255

    mask = cv2.bitwise_or(color_mask, cv2.bitwise_and(edges, color_mask))
    if np.count_nonzero(mask > 0) < MIN_SEGMENT_AREA:
        mask = cv2.bitwise_or(color_mask, edges)

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def find_best_component(*, mask: np.ndarray, roi: np.ndarray, edges: np.ndarray, coarse_box: Box) -> Box | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box: Box | None = None
    best_score = float("-inf")
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width * height < MIN_SEGMENT_AREA:
            continue
        candidate = Box(left=x, top=y, width=width, height=height)
        score = score_segment_candidate(candidate=candidate, coarse_box=coarse_box, roi=roi, edges=edges)
        if score > best_score:
            best_score = score
            best_box = candidate
    return best_box


def score_segment_candidate(*, candidate: Box, coarse_box: Box, roi: np.ndarray, edges: np.ndarray) -> float:
    overlap = iou(candidate, coarse_box)
    aspect_ratio = candidate.width / max(candidate.height, 1)
    squareness = 1.0 - min(abs(aspect_ratio - 1.0), 1.0)
    area_ratio = (candidate.width * candidate.height) / max(roi.shape[0] * roi.shape[1], 1)
    size_score = 1.0 - min(abs(area_ratio - 0.14) / 0.14, 1.0)
    horizontal_distance = abs(candidate.center_x - coarse_box.center_x) / max(roi.shape[1], 1)
    vertical_position = candidate.center_y / max(roi.shape[0], 1)
    upper_preference = 1.0 - min(vertical_position, 1.0)
    edge_region = edges[candidate.top : candidate.bottom, candidate.left : candidate.right]
    edge_density = np.count_nonzero(edge_region > 0) / max(candidate.width * candidate.height, 1)
    color_region = roi[candidate.top : candidate.bottom, candidate.left : candidate.right]
    color_std = float(np.std(color_region)) / 64.0
    return (
        overlap * 0.9
        + squareness * 0.9
        + size_score * 1.1
        + upper_preference * 0.8
        + min(edge_density * 8.0, 1.2)
        + min(color_std, 1.0)
        - horizontal_distance * 0.35
    )


def trim_component_mask(*, mask: np.ndarray, component_box: Box) -> Box:
    component = mask[component_box.top : component_box.bottom, component_box.left : component_box.right]
    rows = np.where(np.count_nonzero(component > 0, axis=1) >= MIN_FOREGROUND_PIXELS_PER_LINE)[0]
    cols = np.where(np.count_nonzero(component > 0, axis=0) >= MIN_FOREGROUND_PIXELS_PER_LINE)[0]
    if rows.size == 0 or cols.size == 0:
        return component_box

    top = component_box.top + max(int(rows[0]) - TRIM_PADDING, 0)
    bottom = component_box.top + min(int(rows[-1]) + TRIM_PADDING + 1, component.shape[0])
    left = component_box.left + max(int(cols[0]) - TRIM_PADDING, 0)
    right = component_box.left + min(int(cols[-1]) + TRIM_PADDING + 1, component.shape[1])
    return Box(left=left, top=top, width=right - left, height=bottom - top)


def save_cv_debug_images(
    *,
    roi: np.ndarray,
    mask: np.ndarray,
    coarse_box: Box,
    refined_box: Box | None,
    output_prefix: Path,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_prefix.with_name(f"{output_prefix.name}_mask.png")), mask)

    overlay = roi.copy()
    cv2.rectangle(
        overlay,
        (coarse_box.left, coarse_box.top),
        (coarse_box.right - 1, coarse_box.bottom - 1),
        (0, 255, 255),
        1,
    )
    if refined_box is not None:
        cv2.rectangle(
            overlay,
            (refined_box.left, refined_box.top),
            (refined_box.right - 1, refined_box.bottom - 1),
            (0, 255, 0),
            2,
        )
    cv2.imwrite(str(output_prefix.with_name(f"{output_prefix.name}_overlay.png")), overlay)


def icon_box_has_signal(icon_box: Box) -> bool:
    return icon_box.width >= MIN_ICON_SIDE and icon_box.height >= MIN_ICON_SIDE


def score_candidate(
    *,
    box: Box,
    image_width: int,
    image_height: int,
    confidence: float,
    matched_text: str | None,
    target_name: str,
) -> float:
    coverage = (box.width * box.height) / (image_width * image_height)
    return confidence + text_match_score(target_name, matched_text) - coverage


def score_leaf_result(
    *,
    box: Box,
    image_width: int,
    image_height: int,
    confidence: float,
    matched_text: str | None,
    target_name: str,
) -> float:
    coverage = (box.width * box.height) / (image_width * image_height)
    return confidence + text_match_score(target_name, matched_text) - (coverage * 0.5)


def choose_best_result(results: list[SearchResult]) -> SearchResult:
    return max(
        results,
        key=lambda result: (result.score, result.planner_support, result.leaf_text_score, result.confidence),
    )


def read_matched_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_text(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum() or character.isspace()).strip()


def text_match_score(target_name: str, matched_text: str | None) -> float:
    if not matched_text:
        return 0.0

    target_clean = normalize_text(target_name)
    matched_clean = normalize_text(matched_text)
    if not target_clean or not matched_clean:
        return 0.0

    target_compact = target_clean.replace(" ", "")
    matched_compact = matched_clean.replace(" ", "")
    if target_compact in matched_compact or matched_compact in target_compact:
        return 0.6

    target_tokens = set(target_clean.split())
    matched_tokens = set(matched_clean.split())
    token_overlap = 0.0
    if target_tokens:
        token_overlap = len(target_tokens & matched_tokens) / len(target_tokens)

    similarity = difflib.SequenceMatcher(None, target_compact, matched_compact).ratio()
    return max(token_overlap, similarity) * 0.6


def apply_nms(candidates: list[Candidate], *, threshold: float) -> list[Candidate]:
    kept: list[Candidate] = []
    for candidate in candidates:
        if all(iou(candidate.box, existing.box) < threshold for existing in kept):
            kept.append(candidate)
    return kept


def iou(first: Box, second: Box) -> float:
    overlap_left = max(first.left, second.left)
    overlap_top = max(first.top, second.top)
    overlap_right = min(first.right, second.right)
    overlap_bottom = min(first.bottom, second.bottom)
    if overlap_right <= overlap_left or overlap_bottom <= overlap_top:
        return 0.0
    overlap_area = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
    first_area = first.width * first.height
    second_area = second.width * second.height
    return overlap_area / (first_area + second_area - overlap_area)


def read_png_dimensions(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    return width, height


def prepare_request_image(
    *,
    source_path: Path,
    output_path: Path,
    max_side: int,
    powershell_executable: str,
) -> tuple[Path, float]:
    width, height = read_png_dimensions(source_path)
    largest_side = max(width, height)
    if largest_side <= max_side:
        return source_path, 1.0

    scale = max_side / largest_side
    resize_image(
        source_path=source_path,
        output_path=output_path,
        width=max(int(round(width * scale)), 1),
        height=max(int(round(height * scale)), 1),
        powershell_executable=powershell_executable,
    )
    return output_path, scale


def crop_image(
    *,
    source_path: Path,
    crop_box: Box,
    output_path: Path,
    powershell_executable: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script = "\n".join(
        [
            "Add-Type -AssemblyName System.Drawing",
            "$source = [System.Drawing.Image]::FromFile('{0}')".format(ps_quote(str(source_path.resolve()))),
            "$bitmap = New-Object System.Drawing.Bitmap {0}, {1}".format(crop_box.width, crop_box.height),
            "$graphics = [System.Drawing.Graphics]::FromImage($bitmap)",
            "$sourceRect = New-Object System.Drawing.Rectangle {0}, {1}, {2}, {3}".format(crop_box.left, crop_box.top, crop_box.width, crop_box.height),
            "$destRect = New-Object System.Drawing.Rectangle 0, 0, {0}, {1}".format(crop_box.width, crop_box.height),
            "$graphics.DrawImage($source, $destRect, $sourceRect, [System.Drawing.GraphicsUnit]::Pixel)",
            "$bitmap.Save('{0}', [System.Drawing.Imaging.ImageFormat]::Png)".format(
                ps_quote(str(output_path.resolve()))
            ),
            "$graphics.Dispose()",
            "$bitmap.Dispose()",
            "$source.Dispose()",
        ]
    )
    result = subprocess.run(
        [powershell_executable, "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "crop failed")


def resize_image(
    *,
    source_path: Path,
    output_path: Path,
    width: int,
    height: int,
    powershell_executable: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script = "\n".join(
        [
            "Add-Type -AssemblyName System.Drawing",
            "$source = [System.Drawing.Image]::FromFile('{0}')".format(ps_quote(str(source_path.resolve()))),
            "$bitmap = New-Object System.Drawing.Bitmap {0}, {1}".format(width, height),
            "$graphics = [System.Drawing.Graphics]::FromImage($bitmap)",
            "$graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic",
            "$graphics.DrawImage($source, 0, 0, {0}, {1})".format(width, height),
            "$bitmap.Save('{0}', [System.Drawing.Imaging.ImageFormat]::Png)".format(
                ps_quote(str(output_path.resolve()))
            ),
            "$graphics.Dispose()",
            "$bitmap.Dispose()",
            "$source.Dispose()",
        ]
    )
    result = subprocess.run(
        [powershell_executable, "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "resize failed")


def annotate_box(
    *,
    screenshot_path: Path,
    box: Box,
    caption: str,
    output_path: Path,
    powershell_executable: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script = "\n".join(
        [
            "Add-Type -AssemblyName System.Drawing",
            "$source = [System.Drawing.Image]::FromFile('{0}')".format(ps_quote(str(screenshot_path.resolve()))),
            "$bitmap = New-Object System.Drawing.Bitmap $source",
            "$source.Dispose()",
            "$graphics = [System.Drawing.Graphics]::FromImage($bitmap)",
            "$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias",
            "$pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::Lime), 3",
            "$centerPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::Red), 2",
            "$graphics.DrawRectangle($pen, {0}, {1}, {2}, {3})".format(box.left, box.top, box.width, box.height),
            "$graphics.DrawLine($centerPen, {0}, {1}, {2}, {1})".format(box.center_x - 8, box.center_y, box.center_x + 8),
            "$graphics.DrawLine($centerPen, {0}, {1}, {0}, {2})".format(box.center_x, box.center_y - 8, box.center_y + 8),
            "$font = New-Object System.Drawing.Font('Segoe UI', 12, [System.Drawing.FontStyle]::Bold)",
            "$caption = '{0}'".format(ps_quote(caption)),
            "$measure = $graphics.MeasureString($caption, $font)",
            "$labelWidth = [int][Math]::Ceiling($measure.Width) + 12",
            "$labelHeight = [int][Math]::Ceiling($measure.Height) + 8",
            "$labelTop = [Math]::Max(0, {0} - $labelHeight - 4)".format(box.top),
            "$labelBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(180, 0, 0, 0))",
            "$graphics.FillRectangle($labelBrush, {0}, $labelTop, $labelWidth, $labelHeight)".format(box.left),
            "$graphics.DrawString($caption, $font, [System.Drawing.Brushes]::White, [float]({0} + 6), [float]($labelTop + 4))".format(box.left),
            "$bitmap.Save('{0}', [System.Drawing.Imaging.ImageFormat]::Png)".format(ps_quote(str(output_path.resolve()))),
            "$graphics.Dispose()",
            "$pen.Dispose()",
            "$centerPen.Dispose()",
            "$labelBrush.Dispose()",
            "$font.Dispose()",
            "$bitmap.Dispose()",
        ]
    )
    result = subprocess.run(
        [powershell_executable, "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "annotate failed")


def ps_quote(value: str) -> str:
    return value.replace("'", "''")


def show_desktop(powershell_executable: str) -> None:
    subprocess.run(
        [
            powershell_executable,
            "-NoProfile",
            "-Command",
            "(New-Object -ComObject Shell.Application).MinimizeAll()",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    sleep(0.4)


def capture_desktop_screenshot(*, output_path: Path, powershell_executable: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script = "\n".join(
        [
            'Add-Type @"',
            "using System;",
            "using System.Runtime.InteropServices;",
            "public static class NativeMethods {",
            '    [DllImport("user32.dll")]',
            "    public static extern bool SetProcessDPIAware();",
            "}",
            '"@',
            "[void][NativeMethods]::SetProcessDPIAware()",
            "Add-Type -AssemblyName System.Windows.Forms",
            "Add-Type -AssemblyName System.Drawing",
            "$bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen",
            "$bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height",
            "$graphics = [System.Drawing.Graphics]::FromImage($bitmap)",
            "$graphics.CopyFromScreen($bounds.Left, $bounds.Top, 0, 0, $bitmap.Size)",
            "$bitmap.Save('{0}', [System.Drawing.Imaging.ImageFormat]::Png)".format(
                ps_quote(str(output_path.resolve()))
            ),
            "$graphics.Dispose()",
            "$bitmap.Dispose()",
        ]
    )
    result = subprocess.run(
        [powershell_executable, "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "screenshot failed")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def normalize_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized[: -len("/chat/completions")]
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grounding_algorithm",
        description="Run a simple ScreenSeekeR-style recursive icon grounding search.",
    )
    parser.add_argument("--target", required=True, help="Target icon or app name, like notepad or vscode.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("grounding_algorithm") / "output",
        help="Directory for cropped patches and the final annotated image.",
    )
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH, help="Maximum recursive search depth.")
    parser.add_argument(
        "--leaf-size",
        type=int,
        default=LEAF_IMAGE_MAX_SIDE,
        help="Use direct grounding when a patch is this size or smaller.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path.cwd() / ".env")
    args = build_parser().parse_args(argv)
    planner_base_url = os.getenv("TJM_PLANNER_BASE_URL", DEFAULT_BASE_URL)
    planner_api_key = os.environ["TJM_PLANNER_API_KEY"]
    planner_model = os.environ["TJM_PLANNER_MODEL"]
    grounding_base_url = os.getenv("TJM_VLM_BASE_URL", DEFAULT_BASE_URL)
    grounding_api_key = os.environ["TJM_VLM_API_KEY"]
    grounding_model = os.environ["TJM_VLM_MODEL"]
    seeker = ScreenSeeker(
        planner_base_url=planner_base_url,
        planner_api_key=planner_api_key,
        planner_model=planner_model,
        grounding_base_url=grounding_base_url,
        grounding_api_key=grounding_api_key,
        grounding_model=grounding_model,
        output_dir=args.output_dir,
        max_depth=args.max_depth,
        leaf_size=args.leaf_size,
    )
    result = seeker.locate(target_name=args.target)
    print(f"Target: {result.target}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Depth: {result.depth}")
    print(
        "Box: "
        f"left={result.box.left}, top={result.box.top}, "
        f"width={result.box.width}, height={result.box.height}"
    )
    print(f"Annotated image: {result.annotated_path}")
    return 0
