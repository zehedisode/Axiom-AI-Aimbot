from __future__ import annotations

import math
import queue
import time
from typing import TYPE_CHECKING, Dict, List, Tuple

import win32api

if TYPE_CHECKING:
    from .ai_loop_state import LoopState
    from .config import Config


def update_crosshair_position(
    config: Config, half_width: int, half_height: int
) -> None:
    if config.fov_follow_mouse:
        try:
            x, y = win32api.GetCursorPos()
            config.crosshairX, config.crosshairY = x, y
        except (OSError, RuntimeError):
            config.crosshairX, config.crosshairY = half_width, half_height
    else:
        config.crosshairX, config.crosshairY = half_width, half_height


def clear_queues(boxes_queue: queue.Queue, confidences_queue: queue.Queue) -> None:
    try:
        while not boxes_queue.empty():
            boxes_queue.get_nowait()
        while not confidences_queue.empty():
            confidences_queue.get_nowait()
    except queue.Empty:
        pass
    boxes_queue.put([])
    confidences_queue.put([])


def calculate_detection_region(
    config: Config, crosshair_x: int, crosshair_y: int
) -> Dict[str, int]:
    detection_size = int(getattr(config, "detect_range_size", config.height))
    detection_size = max(int(config.fov_size), min(int(config.height), detection_size))
    half_detection_size = detection_size // 2

    region_left = max(0, crosshair_x - half_detection_size)
    region_top = max(0, crosshair_y - half_detection_size)
    region_width = max(0, min(detection_size, config.width - region_left))
    region_height = max(0, min(detection_size, config.height - region_top))

    return {
        "left": region_left,
        "top": region_top,
        "width": region_width,
        "height": region_height,
    }


def filter_boxes_by_fov(
    boxes: List[List[float]],
    confidences: List[float],
    crosshair_x: int,
    crosshair_y: int,
    fov_size: int,
) -> Tuple[List[List[float]], List[float]]:
    if not boxes:
        return [], []

    fov_half = fov_size // 2
    fov_margin = max(15, fov_half // 6)
    fov_left = crosshair_x - fov_half - fov_margin
    fov_top = crosshair_y - fov_half - fov_margin
    fov_right = crosshair_x + fov_half + fov_margin
    fov_bottom = crosshair_y + fov_half + fov_margin

    filtered_boxes = []
    filtered_confidences = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        is_small = box_w < 30 and box_h < 60

        if is_small:
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            if fov_left <= cx <= fov_right and fov_top <= cy <= fov_bottom:
                filtered_boxes.append(box)
                if i < len(confidences):
                    filtered_confidences.append(confidences[i])
        else:
            if x1 < fov_right and x2 > fov_left and y1 < fov_bottom and y2 > fov_top:
                filtered_boxes.append(box)
                if i < len(confidences):
                    filtered_confidences.append(confidences[i])

    return filtered_boxes, filtered_confidences


def _box_iou(a: List[float], b: List[float]) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _smooth_box(
    prev: List[float], curr: List[float], alpha: float = 0.4
) -> List[float]:
    return [
        prev[0] * alpha + curr[0] * (1 - alpha),
        prev[1] * alpha + curr[1] * (1 - alpha),
        prev[2] * alpha + curr[2] * (1 - alpha),
        prev[3] * alpha + curr[3] * (1 - alpha),
    ]


def apply_temporal_filter(
    boxes: List[List[float]],
    confidences: List[float],
    state: LoopState,
    current_time: float,
    confirm_frames: int = 1,
    expire_time: float = 0.5,
) -> Tuple[List[List[float]], List[float]]:
    if not boxes:
        elapsed = current_time - state.target_last_seen_time
        if elapsed > expire_time:
            state.target_confirm_count = 0
            state.target_last_box = None
            state.smoothed_box = None
        return [], []

    best_box = boxes[0]
    best_conf = confidences[0]

    if state.target_last_box is not None:
        best_iou = 0.0
        best_match_idx = 0
        for i, box in enumerate(boxes):
            iou = _box_iou(list(state.target_last_box), box)
            if iou > best_iou:
                best_iou = iou
                best_match_idx = i

        if best_iou > 0.05:
            best_box = boxes[best_match_idx]
            best_conf = confidences[best_match_idx]
        else:
            best_box = boxes[0]
            best_conf = confidences[0]

    current_box_tuple = tuple(best_box)

    if state.target_last_box is not None:
        iou_with_last = _box_iou(list(state.target_last_box), best_box)
        if iou_with_last > 0.05:
            state.target_confirm_count = min(
                state.target_confirm_count + 1,
                confirm_frames + 10,
            )
        else:
            state.target_confirm_count = max(1, state.target_confirm_count - 1)
    else:
        state.target_confirm_count = 1

    state.target_last_box = current_box_tuple
    state.target_last_seen_time = current_time

    if state.smoothed_box is not None:
        iou_check = _box_iou(state.smoothed_box, best_box)
        if iou_check > 0.05:
            box_area = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
            alpha = 0.5 if box_area < 800 else 0.4
            best_box = _smooth_box(state.smoothed_box, best_box, alpha=alpha)

    state.smoothed_box = list(best_box)

    if state.target_confirm_count < confirm_frames:
        return [], []

    return [list(best_box)], [best_conf]


def find_closest_target(
    boxes: List[List[float]],
    confidences: List[float],
    crosshair_x: int,
    crosshair_y: int,
    aim_part: str = "head",
    head_height_ratio: float = 0.26,
) -> Tuple[List[List[float]], List[float]]:
    if not boxes:
        return [], []

    closest_box = None
    min_distance_sq = float("inf")
    closest_confidence = 0.5

    for i, box in enumerate(boxes):
        abs_x1, abs_y1, abs_x2, abs_y2 = box
        box_w = abs_x2 - abs_x1
        box_h = abs_y2 - abs_y1
        box_center_x = (abs_x1 + abs_x2) * 0.5

        if aim_part == "head":
            head_pixel_h = box_h * head_height_ratio
            target_y = abs_y1 + head_pixel_h * 0.35
            target_x = box_center_x
        else:
            target_x = box_center_x
            target_y = (abs_y1 + abs_y2) * 0.5

        dx = target_x - crosshair_x
        dy = target_y - crosshair_y
        distance_sq = dx * dx + dy * dy

        if distance_sq < min_distance_sq:
            min_distance_sq = distance_sq
            closest_box = box
            closest_confidence = confidences[i] if i < len(confidences) else 0.5

    if closest_box:
        return [closest_box], [closest_confidence]
    return [], []


def update_queues(
    overlay_boxes_queue: queue.Queue,
    overlay_confidences_queue: queue.Queue,
    boxes: List[List[float]],
    confidences: List[float],
    auto_fire_queue: queue.Queue | None = None,
) -> None:
    try:
        if overlay_boxes_queue.full():
            overlay_boxes_queue.get_nowait()
        if overlay_confidences_queue.full():
            overlay_confidences_queue.get_nowait()
    except queue.Empty:
        pass

    overlay_boxes_queue.put(boxes)
    overlay_confidences_queue.put(confidences)

    if auto_fire_queue is not None:
        try:
            if auto_fire_queue.full():
                auto_fire_queue.get_nowait()
        except queue.Empty:
            pass
        auto_fire_queue.put(list(boxes))
