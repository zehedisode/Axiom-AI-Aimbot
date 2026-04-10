import numpy as np
from typing import Tuple


class SmartTracker:
    def __init__(
        self,
        smoothing_factor: float = 0.5,
        stop_threshold: float = 20.0,
        position_deadzone: float = 3.0,
        max_prediction_pixels: float = 40.0,
    ):
        self.alpha = smoothing_factor
        self.stop_threshold = stop_threshold
        self.position_deadzone = position_deadzone
        self.max_prediction_pixels = max_prediction_pixels

        self.last_x: float | None = None
        self.last_y: float | None = None
        self.vx = 0.0
        self.vy = 0.0
        self.initialized = False

    def update(
        self, measured_x: float, measured_y: float, dt: float
    ) -> Tuple[float, float, float, float]:
        if not self.initialized or dt <= 0:
            self.last_x = measured_x
            self.last_y = measured_y
            self.vx = 0.0
            self.vy = 0.0
            self.initialized = True
            return measured_x, measured_y, 0.0, 0.0

        raw_vx = (measured_x - self.last_x) / dt
        raw_vy = (measured_y - self.last_y) / dt

        dot_product = raw_vx * self.vx + raw_vy * self.vy
        if dot_product < 0:
            self.vx = raw_vx
            self.vy = raw_vy
        else:
            self.vx = self.vx * self.alpha + raw_vx * (1 - self.alpha)
            self.vy = self.vy * self.alpha + raw_vy * (1 - self.alpha)

        if abs(self.vx) < self.stop_threshold:
            self.vx = 0
        if abs(self.vy) < self.stop_threshold:
            self.vy = 0

        self.last_x = measured_x
        self.last_y = measured_y

        return measured_x, measured_y, self.vx, self.vy

    def get_predicted_position(self, prediction_time: float) -> Tuple[float, float]:
        if not self.initialized:
            return 0.0, 0.0

        if self.last_x is None or self.last_y is None:
            return 0.0, 0.0

        speed = np.sqrt(self.vx**2 + self.vy**2)

        # For slow/stationary targets, return raw position (no prediction offset)
        # Higher threshold (3x) prevents aiming offset on nearly-still targets
        if speed < self.stop_threshold * 3:
            return self.last_x, self.last_y

        pred_x = self.last_x + self.vx * prediction_time
        pred_y = self.last_y + self.vy * prediction_time

        # Clamp prediction to max distance to prevent overshoot jitter
        dx = pred_x - self.last_x
        dy = pred_y - self.last_y
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > self.max_prediction_pixels:
            scale = self.max_prediction_pixels / dist
            pred_x = self.last_x + dx * scale
            pred_y = self.last_y + dy * scale

        return pred_x, pred_y

    def reset(self):
        self.last_x = None
        self.last_y = None
        self.vx = 0.0
        self.vy = 0.0
        self.initialized = False
