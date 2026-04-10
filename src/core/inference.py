# inference.py
"""AI inference module - Image preprocessing, post-processing, and PID controller"""

from __future__ import annotations

import math
from typing import List, Tuple, Any

import cv2
import numpy as np
import numpy.typing as npt


class PIDController:
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

        # Dead zone: suppress sub-pixel noise (prevents micro-jitter)
        if dist < 1.0:
            self.integral *= 0.5
            self.previous_error = error
            return 0.0

        # Near-target precision mode: smoothly reduce gains to prevent jitter
        # while still allowing fine adjustments
        if dist < 10.0:
            # Smooth quadratic fade — gentler as we get closer
            t = (dist - 1.0) / 9.0  # 0.0 at 1px, 1.0 at 10px
            scale = 0.25 + 0.75 * (t * t)  # quadratic ease-in
            effective_kp = self.Kp * scale
            effective_kd = self.Kd * (0.6 + 0.4 * t)
            derivative = error - self.previous_error
            # Don't aggressively decay integral near target — causes drift
            self.integral = self.integral * 0.95 + error * 0.05
            self.integral = max(-200.0, min(200.0, self.integral))
            output = (
                effective_kp * error
                + self.Ki * self.integral
                + effective_kd * derivative
            )
            self.previous_error = error
            return output

        # Full power mode — target is far away
        self.integral += error
        self.integral = max(-500.0, min(500.0, self.integral))

        derivative = error - self.previous_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        self.previous_error = error
        return output


def _letterbox(
    image: npt.NDArray[np.uint8], target_size: int
) -> Tuple[npt.NDArray[np.uint8], float, Tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_right = target_size - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return padded, scale, (pad_top, pad_left, pad_bottom, pad_right)


def _apply_clahe(
    image: npt.NDArray[np.uint8], clip_limit: float = 2.0, tile_grid: int = 8
) -> npt.NDArray[np.uint8]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_ch = clahe.apply(l_ch)
    merged = cv2.merge([l_ch, a_ch, b_ch])
    return np.array(cv2.cvtColor(merged, cv2.COLOR_LAB2BGR), dtype=np.uint8)


def _gamma_correct(
    image: npt.NDArray[np.uint8], gamma: float = 1.0
) -> npt.NDArray[np.uint8]:
    if abs(gamma - 1.0) < 0.01:
        return image
    inv_gamma = 1.0 / gamma
    lut = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(image, lut)


def _apply_sharpen(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Light unsharp mask to restore edge detail after resize for small/distant targets."""
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _estimate_target_size(image: npt.NDArray[np.uint8], model_input_size: int) -> float:
    """Heuristic: estimate whether enhancement is needed for better detection.

    Uses a lightweight variance-of-Laplacian check instead of full Canny
    edge detection to reduce preprocessing overhead by ~60%.

    Returns a rough ratio (0.0-1.0). Low ratio = distant targets, needs enhancement.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Downsample for speed — we only need a rough estimate
    h, w = gray.shape[:2]
    max_dim = 160
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))))
    else:
        small = gray

    # Variance of Laplacian is much faster than Canny and correlates with detail
    laplacian_var = cv2.Laplacian(small, cv2.CV_64F).var()

    # Normalize to a 0-1 range (empirically, variance > 500 means rich detail)
    normalized = min(laplacian_var / 500.0, 1.0)
    return normalized


def preprocess_image(
    image: npt.NDArray[np.uint8], model_input_size: int
) -> npt.NDArray[np.float32]:
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    h, w = image.shape[:2]
    is_small_frame = h < model_input_size or w < model_input_size
    edge_ratio = _estimate_target_size(image, model_input_size)
    needs_enhancement = is_small_frame or edge_ratio < 0.20

    if needs_enhancement:
        image = _apply_clahe(image, clip_limit=2.5, tile_grid=4)
        image = _gamma_correct(image, gamma=0.9)

    if image.shape[0] != model_input_size or image.shape[1] != model_input_size:
        if needs_enhancement:
            image = cv2.resize(
                image,
                (model_input_size, model_input_size),
                interpolation=cv2.INTER_LINEAR,
            )
            image = _apply_sharpen(image)
        else:
            image = cv2.resize(
                image,
                (model_input_size, model_input_size),
                interpolation=cv2.INTER_AREA,
            )

    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0 / 255.0,
        size=(model_input_size, model_input_size),
        swapRB=True,
        crop=False,
    )

    return np.ascontiguousarray(blob, dtype=np.float32)


def preprocess_image_zoom(
    image: npt.NDArray[np.uint8], model_input_size: int, zoom_factor: float = 2.0
) -> Tuple[npt.NDArray[np.float32], float, float, int, int]:
    """Preprocess with center crop zoom for better distant target detection.

    Crops the center portion of the image (1/zoom_factor of each dimension),
    then upscales to model_input_size. This makes small/distant targets appear
    larger in the model input, improving detection accuracy.

    Args:
        image: Input frame.
        model_input_size: Model input dimension (e.g., 640).
        zoom_factor: Zoom level (1.0 = no zoom, 2.0 = 2x center crop, etc.)

    Returns:
        blob: Preprocessed input tensor (1, 3, H, W).
        scale_x: X scale factor from crop coordinates to original coordinates.
        scale_y: Y scale factor from crop coordinates to original coordinates.
        crop_x: X offset of the crop in original image.
        crop_y: Y offset of the crop in original image.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    h, w = image.shape[:2]
    zoom_factor = max(1.0, min(zoom_factor, 4.0))

    # Calculate crop region (center of image)
    crop_w = int(w / zoom_factor)
    crop_h = int(h / zoom_factor)
    crop_x = (w - crop_w) // 2
    crop_y = (h - crop_h) // 2

    # Clamp crop to image bounds
    crop_x = max(0, crop_x)
    crop_y = max(0, crop_y)
    crop_w = min(crop_w, w - crop_x)
    crop_h = min(crop_h, h - crop_y)

    # Crop
    cropped = image[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

    # Enhance small/distant targets
    is_small_frame = crop_h < model_input_size or crop_w < model_input_size
    if is_small_frame:
        cropped = _apply_clahe(cropped, clip_limit=2.5, tile_grid=4)
        cropped = _gamma_correct(cropped, gamma=0.9)

    # Resize to model input
    if is_small_frame:
        cropped = cv2.resize(
            cropped,
            (model_input_size, model_input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        cropped = _apply_sharpen(cropped)
    else:
        cropped = cv2.resize(
            cropped,
            (model_input_size, model_input_size),
            interpolation=cv2.INTER_AREA,
        )

    blob = cv2.dnn.blobFromImage(
        cropped,
        scalefactor=1.0 / 255.0,
        size=(model_input_size, model_input_size),
        swapRB=True,
        crop=False,
    )

    return np.ascontiguousarray(blob, dtype=np.float32), crop_x, crop_y, crop_w, crop_h


def preprocess_image_letterbox(
    image: npt.NDArray[np.uint8], model_input_size: int
) -> Tuple[npt.NDArray[np.float32], float, Tuple[int, int, int, int]]:
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    h, w = image.shape[:2]
    is_small_frame = h < model_input_size or w < model_input_size
    edge_ratio = _estimate_target_size(image, model_input_size)
    needs_enhancement = is_small_frame or edge_ratio < 0.20

    if needs_enhancement:
        image = _apply_clahe(image, clip_limit=2.5, tile_grid=4)
        image = _gamma_correct(image, gamma=0.9)

    lb_image, scale, padding = _letterbox(image, model_input_size)

    if needs_enhancement:
        lb_image = _apply_sharpen(lb_image)

    blob = cv2.dnn.blobFromImage(
        lb_image,
        scalefactor=1.0 / 255.0,
        size=(model_input_size, model_input_size),
        swapRB=True,
        crop=False,
    )

    return np.ascontiguousarray(blob, dtype=np.float32), scale, padding


def postprocess_outputs(
    outputs: List[Any],
    original_width: int,
    original_height: int,
    model_input_size: int,
    min_confidence: float,
    offset_x: int = 0,
    offset_y: int = 0,
    min_box_area_ratio: float = 0.0001,
    letterbox_scale: float = 0.0,
    letterbox_padding: Tuple[int, int, int, int] | None = None,
) -> Tuple[List[List[float]], List[float]]:
    predictions = outputs[0][0].T

    conf_mask = predictions[:, 4] >= min_confidence
    filtered_predictions = predictions[conf_mask]

    if len(filtered_predictions) == 0:
        return [], []

    cx, cy, w, h = (
        filtered_predictions[:, 0],
        filtered_predictions[:, 1],
        filtered_predictions[:, 2],
        filtered_predictions[:, 3],
    )

    if letterbox_scale > 0.0 and letterbox_padding is not None:
        pad_top, pad_left, pad_bottom, pad_right = letterbox_padding
        cx = (cx - pad_left) / letterbox_scale
        cy = (cy - pad_top) / letterbox_scale
        w = w / letterbox_scale
        h = h / letterbox_scale

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        x1 = np.clip(x1, 0, original_width)
        y1 = np.clip(y1, 0, original_height)
        x2 = np.clip(x2, 0, original_width)
        y2 = np.clip(y2, 0, original_height)

        x1 += offset_x
        y1 += offset_y
        x2 += offset_x
        y2 += offset_y
    else:
        scale_x = original_width / model_input_size
        scale_y = original_height / model_input_size

        x1 = (cx - w / 2) * scale_x + offset_x
        y1 = (cy - h / 2) * scale_y + offset_y
        x2 = (cx + w / 2) * scale_x + offset_x
        y2 = (cy + h / 2) * scale_y + offset_y

    box_ws = x2 - x1
    box_hs = y2 - y1
    box_areas = box_ws * box_hs
    image_area = original_width * original_height
    min_box_area = image_area * min_box_area_ratio

    min_box_dim = 6.0

    aspect_ratio = box_ws / np.maximum(box_hs, 1.0)

    valid_mask = (
        (box_areas >= min_box_area)
        & (box_ws >= min_box_dim)
        & (box_hs >= min_box_dim)
        & (aspect_ratio > 0.02)
        & (aspect_ratio < 40.0)
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
