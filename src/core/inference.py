# inference.py
"""AI inference module - Image preprocessing, post-processing, and PID controller"""

from __future__ import annotations

import math
from typing import List, Tuple, Any

import cv2
import numpy as np
import numpy.typing as npt


class PIDController:
    """PID Controller with distance-adaptive response and deadzone for lock stability"""

    DEADZONE_PX = 2.0

    def __init__(self, Kp: float, Ki: float, Kd: float) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self._error_distance: float = 100.0
        self.reset()

    def reset(self) -> None:
        self.integral: float = 0.0
        self.previous_error: float = 0.0

    def set_distance_context(self, distance: float) -> None:
        self._error_distance = max(0.0, distance)

    def update(self, error: float) -> float:
        dist = self._error_distance

        if dist < self.DEADZONE_PX:
            self.integral *= 0.5
            self.previous_error = error * 0.3
            return 0.0

        self.integral += error
        self.integral = max(-500.0, min(500.0, self.integral))

        derivative = error - self.previous_error

        adjusted_kp = self._calculate_adaptive_kp(self.Kp)

        if dist > 50.0:
            ki_scale = 1.0
        elif dist > 15.0:
            ki_scale = (dist - 15.0) / 35.0
        else:
            ki_scale = 0.0
            self.integral *= 0.7

        if dist < 8.0:
            kd_boost = 2.5
        elif dist < 20.0:
            kd_boost = 1.5
        else:
            kd_boost = 1.0

        output = (
            (adjusted_kp * error)
            + (self.Ki * ki_scale * self.integral)
            + (self.Kd * kd_boost * derivative)
        )

        self.previous_error = error

        return output

    def _calculate_adaptive_kp(self, kp: float) -> float:
        dist = self._error_distance

        if dist > 80.0:
            return kp * (1.0 + min(kp, 0.5) * 2.0)
        elif dist > 15.0:
            t = (dist - 15.0) / 65.0
            base = kp
            boost = min(kp, 0.5) * 2.0 * t
            return base + boost
        elif dist > 5.0:
            return kp * 0.85
        else:
            return kp * 0.6


def preprocess_image(
    image: npt.NDArray[np.uint8], model_input_size: int
) -> npt.NDArray[np.float32]:
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.shape[0] != model_input_size or image.shape[1] != model_input_size:
        image = cv2.resize(
            image, (model_input_size, model_input_size), interpolation=cv2.INTER_NEAREST
        )

    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0 / 255.0,
        size=(model_input_size, model_input_size),
        swapRB=True,
        crop=False,
    )

    return np.ascontiguousarray(blob, dtype=np.float32)


def postprocess_outputs(
    outputs: List[Any],
    original_width: int,
    original_height: int,
    model_input_size: int,
    min_confidence: float,
    offset_x: int = 0,
    offset_y: int = 0,
    min_box_area_ratio: float = 0.0001,
) -> Tuple[List[List[float]], List[float]]:
    predictions = outputs[0][0].T

    conf_mask = predictions[:, 4] >= min_confidence
    filtered_predictions = predictions[conf_mask]

    if len(filtered_predictions) == 0:
        return [], []

    scale_x = original_width / model_input_size
    scale_y = original_height / model_input_size

    cx, cy, w, h = (
        filtered_predictions[:, 0],
        filtered_predictions[:, 1],
        filtered_predictions[:, 2],
        filtered_predictions[:, 3],
    )

    x1 = (cx - w / 2) * scale_x + offset_x
    y1 = (cy - h / 2) * scale_y + offset_y
    x2 = (cx + w / 2) * scale_x + offset_x
    y2 = (cy + h / 2) * scale_y + offset_y

    box_ws = x2 - x1
    box_hs = y2 - y1
    box_areas = box_ws * box_hs
    image_area = original_width * original_height
    min_box_area = image_area * min_box_area_ratio

    min_box_dim = 8.0

    valid_mask = (
        (box_areas >= min_box_area)
        & (box_ws >= min_box_dim)
        & (box_hs >= min_box_dim)
        & (box_ws / np.maximum(box_hs, 1.0) > 0.08)
        & (box_hs / np.maximum(box_ws, 1.0) > 0.08)
    )

    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return [], []

    x1 = x1[valid_indices]
    y1 = y1[valid_indices]
    x2 = x2[valid_indices]
    y2 = y2[valid_indices]
    confidences = filtered_predictions[valid_indices, 4]

    boxes = np.stack([x1, y1, x2, y2], axis=1).tolist()
    confidences = confidences.tolist()

    return boxes, confidences


def non_max_suppression(
    boxes: List[List[float]], confidences: List[float], iou_threshold: float = 0.45
) -> Tuple[List[List[float]], List[float]]:
    if len(boxes) == 0:
        return [], []

    boxes_arr = np.array(boxes, dtype=np.float64)
    confidences_arr = np.array(confidences, dtype=np.float64)
    areas = (boxes_arr[:, 2] - boxes_arr[:, 0]) * (boxes_arr[:, 3] - boxes_arr[:, 1])
    order = confidences_arr.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        xx1 = np.maximum(boxes_arr[i, 0], boxes_arr[order[1:], 0])
        yy1 = np.maximum(boxes_arr[i, 1], boxes_arr[order[1:], 1])
        xx2 = np.minimum(boxes_arr[i, 2], boxes_arr[order[1:], 2])
        yy2 = np.minimum(boxes_arr[i, 3], boxes_arr[order[1:], 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / np.maximum(union, 1e-6)

        order = order[1:][iou <= iou_threshold]

    return boxes_arr[keep].tolist(), confidences_arr[keep].tolist()
