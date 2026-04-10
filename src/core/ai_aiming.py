from __future__ import annotations

import random
import math
from typing import TYPE_CHECKING, List, Tuple

from win_utils import send_mouse_move

from .ai_loop_state import LoopState
from .inference import PIDController
from .smart_tracker import SmartTracker

if TYPE_CHECKING:
    from .config import Config


def calculate_aim_target(
    box: List[float], aim_part: str, head_height_ratio: float
) -> Tuple[float, float]:
    abs_x1, abs_y1, abs_x2, abs_y2 = box
    box_w, box_h = abs_x2 - abs_x1, abs_y2 - abs_y1
    box_center_x = abs_x1 + box_w * 0.5

    if aim_part == "head":
        target_x = box_center_x
        head_pixel_h = box_h * head_height_ratio
        # Adaptive offset: for small/distant targets, aim at head center (0.45)
        # For large/close targets, aim slightly higher for forehead (0.40)
        if box_h < 60:
            head_offset = 0.45  # Small box = distant, aim center of head region
        elif box_h < 120:
            head_offset = 0.42  # Medium box
        else:
            head_offset = 0.38  # Large box = close, aim upper head
        target_y = abs_y1 + head_pixel_h * head_offset
    elif aim_part == "body":
        target_x = box_center_x
        head_pixel_h = box_h * head_height_ratio
        body_top = abs_y1 + head_pixel_h
        target_y = (body_top + abs_y2) * 0.5
    else:
        target_x = box_center_x
        head_pixel_h = box_h * head_height_ratio
        head_center_y = abs_y1 + head_pixel_h * 0.45
        body_center_y = (abs_y1 + head_pixel_h + abs_y2) * 0.5
        target_y = head_center_y * 0.6 + body_center_y * 0.4

    return target_x, target_y


def _calculate_head_center(
    box: List[float], head_height_ratio: float
) -> Tuple[float, float]:
    abs_x1, abs_y1, abs_x2, abs_y2 = box
    box_w = abs_x2 - abs_x1
    box_h = abs_y2 - abs_y1
    cx = abs_x1 + box_w * 0.5
    head_pixel_h = box_h * head_height_ratio
    # Match the adaptive offset from calculate_aim_target
    if box_h < 60:
        head_offset = 0.45
    elif box_h < 120:
        head_offset = 0.42
    else:
        head_offset = 0.38
    cy = abs_y1 + head_pixel_h * head_offset
    return cx, cy


def _find_best_head_target(
    boxes: List[List[float]],
    crosshair_x: int,
    crosshair_y: int,
    head_height_ratio: float,
) -> Tuple[int, float, float]:
    best_idx = -1
    best_dist_sq = float("inf")
    best_hx = 0.0
    best_hy = 0.0

    for i, box in enumerate(boxes):
        hx, hy = _calculate_head_center(box, head_height_ratio)
        dx = hx - crosshair_x
        dy = hy - crosshair_y
        dist_sq = dx * dx + dy * dy
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_idx = i
            best_hx = hx
            best_hy = hy

    return best_idx, best_hx, best_hy


def process_aiming(
    config: Config,
    boxes: List[List[float]],
    crosshair_x: int,
    crosshair_y: int,
    pid_x: PIDController,
    pid_y: PIDController,
    mouse_method: str,
    state: LoopState,
    current_time: float,
) -> None:
    aim_part = config.aim_part
    head_height_ratio = config.head_height_ratio

    if aim_part == "head":
        best_idx, _, _ = _find_best_head_target(
            boxes, crosshair_x, crosshair_y, head_height_ratio
        )
        if best_idx < 0:
            return
        target_x, target_y = calculate_aim_target(
            boxes[best_idx], aim_part, head_height_ratio
        )
        box = boxes[best_idx]
    else:
        valid_targets = []
        for box in boxes:
            target_x, target_y = calculate_aim_target(box, aim_part, head_height_ratio)
            moveX = target_x - crosshair_x
            moveY = target_y - crosshair_y
            distance_sq = moveX * moveX + moveY * moveY
            valid_targets.append((distance_sq, target_x, target_y, box))

        if not valid_targets:
            return
        valid_targets.sort(key=lambda x: x[0])
        _, target_x, target_y, box = valid_targets[0]

    tracker_enabled = getattr(config, "tracker_enabled", False)
    if tracker_enabled:
        if state.smart_tracker is None:
            state.smart_tracker = SmartTracker(
                smoothing_factor=getattr(config, "tracker_smoothing_factor", 0.5),
                stop_threshold=getattr(config, "tracker_stop_threshold", 20.0),
            )
            state.tracker_last_time = current_time
        else:
            state.smart_tracker.alpha = getattr(config, "tracker_smoothing_factor", 0.5)
            state.smart_tracker.stop_threshold = getattr(
                config, "tracker_stop_threshold", 20.0
            )

        current_box_tuple = tuple(box)
        if state.tracker_last_target_box is not None:
            last_box = state.tracker_last_target_box
            last_cx = (last_box[0] + last_box[2]) * 0.5
            last_cy = (last_box[1] + last_box[3]) * 0.5
            curr_cx = (box[0] + box[2]) * 0.5
            curr_cy = (box[1] + box[3]) * 0.5
            box_distance_sq = (curr_cx - last_cx) ** 2 + (curr_cy - last_cy) ** 2
            if box_distance_sq > 40000:
                state.smart_tracker.reset()
        state.tracker_last_target_box = current_box_tuple

        dt = current_time - state.tracker_last_time
        if dt <= 0:
            dt = 0.01
        state.tracker_last_time = current_time

        state.smart_tracker.update(target_x, target_y, dt)

        prediction_time = getattr(config, "tracker_prediction_time", 0.05)
        pred_x, pred_y = state.smart_tracker.get_predicted_position(prediction_time)

        config.tracker_current_x = target_x
        config.tracker_current_y = target_y
        config.tracker_predicted_x = pred_x
        config.tracker_predicted_y = pred_y
        config.tracker_has_prediction = True

        target_x, target_y = pred_x, pred_y
    else:
        config.tracker_has_prediction = False
        if state.smart_tracker is not None:
            state.smart_tracker.reset()
            state.smart_tracker = None

    errorX = target_x - crosshair_x
    errorY = target_y - crosshair_y

    error_distance = math.sqrt(errorX * errorX + errorY * errorY)

    # Bezier: only when NOT near-locked and far from target
    if getattr(config, "bezier_curve_enabled", False):
        if error_distance > 6.0:
            if not state.target_locked:
                state.target_locked = False
                if (
                    not hasattr(state, "_bezier_needs_new_scalar")
                    or state._bezier_needs_new_scalar
                ):
                    state.bezier_curve_scalar = random.uniform(-1.0, 1.0)
                    state._bezier_needs_new_scalar = False

            strength = float(getattr(config, "bezier_curve_strength", 0.08))
            fade = min(1.0, error_distance / 150.0)
            effective_strength = strength * fade

            perp_x = -errorY
            perp_y = errorX

            errorX += perp_x * effective_strength * state.bezier_curve_scalar
            errorY += perp_y * effective_strength * state.bezier_curve_scalar
        else:
            # Within 6px: disable bezier, aim directly at target
            state.target_locked = True
            state._bezier_needs_new_scalar = True
    else:
        state.target_locked = error_distance <= 2.0

    pid_x.set_distance_context(error_distance)
    pid_y.set_distance_context(error_distance)

    dx, dy = pid_x.update(errorX), pid_y.update(errorY)

    if getattr(config, "aim_y_reduce_enabled", False) and state.aiming_start_time > 0:
        aim_duration = current_time - state.aiming_start_time
        delay = getattr(config, "aim_y_reduce_delay", 0.6)
        if aim_duration > delay:
            fade_out = max(0.0, 1.0 - (aim_duration - delay) / 0.3)
            dy *= fade_out

    move_x, move_y = int(round(dx)), int(round(dy))

    if move_x != 0 or move_y != 0:
        send_mouse_move(move_x, move_y, method=mouse_method)
