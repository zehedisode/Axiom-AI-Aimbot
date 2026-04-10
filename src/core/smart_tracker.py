"""Enhanced target tracking with Kalman filter, Sticky Aim, and EMA smoothing.

Ported techniques from Aimmy V2 (Babyhamsta/Aimmy):
- KalmanTracker: Full 4-state Kalman filter [x, y, vx, vy] from PredictionManager.cs
- StickyAimManager: Target persistence with grace period and velocity prediction from AIManager.cs
- EMASmoother: Output smoothing from MouseManager.cs
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Kalman Tracker (from Aimmy V2 PredictionManager.cs KalmanPrediction)
# ---------------------------------------------------------------------------


class KalmanTracker:
    """Full 4-state Kalman filter for target position prediction.

    State vector: [x, y, vx, vy]
    Measurement: [x, y]

    The Kalman filter adaptively blends predictions with measurements based
    on estimated uncertainty (covariance), giving smoother and more accurate
    predictions than simple EMA smoothing.
    """

    def __init__(
        self,
        process_noise: float = 0.001,
        measurement_noise: float = 0.01,
        velocity_clamp: float = 2000.0,
    ):
        # State: [x, y, vx, vy]
        self.state = np.zeros((4, 1), dtype=np.float64)
        # Covariance matrix — large initial uncertainty
        self.P = np.eye(4, dtype=np.float64) * 500.0

        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.velocity_clamp = velocity_clamp

        self.initialized = False

        # Measurement matrix H — we observe position only
        self.H = np.zeros((2, 4), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

    def predict(self, dt: float) -> Tuple[float, float]:
        """Predict next state (time update). Returns predicted (x, y)."""
        if not self.initialized or dt <= 0:
            return 0.0, 0.0

        # State transition matrix F
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = dt  # x += vx * dt
        F[1, 3] = dt  # y += vy * dt

        # Process noise Q — higher on velocity components
        Q = np.eye(4, dtype=np.float64) * self.process_noise
        Q[2, 2] *= 10.0
        Q[3, 3] *= 10.0

        # Predict: x_pred = F * x, P_pred = F * P * F^T + Q
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

        # Clamp velocity
        self.state[2, 0] = np.clip(
            self.state[2, 0], -self.velocity_clamp, self.velocity_clamp
        )
        self.state[3, 0] = np.clip(
            self.state[3, 0], -self.velocity_clamp, self.velocity_clamp
        )

        return float(self.state[0, 0]), float(self.state[1, 0])

    def update(self, measured_x: float, measured_y: float) -> Tuple[float, float]:
        """Update state with measurement. Returns filtered (x, y)."""
        if not self.initialized:
            self.state[0, 0] = measured_x
            self.state[1, 0] = measured_y
            self.initialized = True
            return measured_x, measured_y

        z = np.array([[measured_x], [measured_y]], dtype=np.float64)

        # Measurement noise R
        R = np.eye(2, dtype=np.float64) * self.measurement_noise

        # Innovation: y = z - H * x_pred
        innovation = z - self.H @ self.state

        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + R

        # Kalman gain: K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.state = self.state + K @ innovation
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P

        return float(self.state[0, 0]), float(self.state[1, 0])

    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate."""
        return float(self.state[2, 0]), float(self.state[3, 0])

    def get_predicted_position(self, prediction_time: float) -> Tuple[float, float]:
        """Get predicted position after prediction_time seconds."""
        if not self.initialized:
            return 0.0, 0.0

        speed = math.sqrt(self.state[2, 0] ** 2 + self.state[3, 0] ** 2)

        # For slow/stationary targets, return raw position (no prediction offset)
        if speed < 50.0:  # ~50 pixels/sec threshold
            return float(self.state[0, 0]), float(self.state[1, 0])

        pred_x = float(self.state[0, 0] + self.state[2, 0] * prediction_time)
        pred_y = float(self.state[1, 0] + self.state[3, 0] * prediction_time)

        return pred_x, pred_y

    def reset(self):
        """Reset tracker state."""
        self.state = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 500.0
        self.initialized = False


# ---------------------------------------------------------------------------
# Legacy SmartTracker (kept for backward compatibility)
# ---------------------------------------------------------------------------


class SmartTracker:
    """Legacy EMA-based velocity tracker.

    Kept as a fallback option. For new behavior use KalmanTracker.
    """

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

        raw_vx = (measured_x - (self.last_x or 0.0)) / dt
        raw_vy = (measured_y - (self.last_y or 0.0)) / dt

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

        lx: float = self.last_x
        ly: float = self.last_y

        speed = np.sqrt(self.vx**2 + self.vy**2)

        if speed < self.stop_threshold * 3:
            return lx, ly

        pred_x = lx + self.vx * prediction_time
        pred_y = ly + self.vy * prediction_time

        dx = pred_x - lx
        dy = pred_y - ly
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > self.max_prediction_pixels:
            scale = self.max_prediction_pixels / dist
            pred_x = lx + dx * scale
            pred_y = ly + dy * scale

        return pred_x, pred_y

    def reset(self):
        self.last_x = None
        self.last_y = None
        self.vx = 0.0
        self.vy = 0.0
        self.initialized = False


# ---------------------------------------------------------------------------
# Sticky Aim Manager (from Aimmy V2 AIManager.cs HandleStickyAim)
# ---------------------------------------------------------------------------


@dataclass
class _TrackedTarget:
    """Internal tracked target state."""

    box: List[float] = field(default_factory=lambda: [0, 0, 0, 0])
    confidence: float = 0.0
    frames_without_detection: int = 0


class StickyAimManager:
    """Target persistence system with grace period and velocity prediction.

    Ported from Aimmy V2's HandleStickyAim (AIManager.cs).

    Key behaviors:
    - **Grace period**: When target is lost, extrapolates position for up to
      MAX_FRAMES_WITHOUT_TARGET frames using velocity, preventing aim flicker.
    - **Size-aware tracking**: Larger tracking radius for bigger targets.
    - **Size similarity**: Won't jump to a differently-sized detection (rejects
      false positives).
    - **Hysteresis**: Requires 3 consecutive frames on a new target before
      switching, preventing rapid target toggling.
    - **Lock score**: Accumulated stickiness that grows while tracking and
      decays when the target is lost.
    """

    MAX_FRAMES_WITHOUT_TARGET: int = 3
    LOCK_SCORE_DECAY: float = 0.85
    LOCK_SCORE_GAIN: float = 15.0
    MAX_LOCK_SCORE: float = 100.0
    REFERENCE_TARGET_SIZE: float = 10000.0
    SIZE_RATIO_THRESHOLD: float = 0.5
    SWITCH_FRAMES: int = 3

    def __init__(self) -> None:
        self._target: Optional[_TrackedTarget] = None
        self._vel_x: float = 0.0
        self._vel_y: float = 0.0
        self._lock_score: float = 0.0
        self._frames_without_match: int = 0

    # ---- public API --------------------------------------------------------

    def update(
        self,
        boxes: List[List[float]],
        confidences: List[float],
        crosshair_x: int,
        crosshair_y: int,
        enabled: bool = True,
    ) -> Tuple[List[List[float]], List[float]]:
        """Run sticky aim logic and return the target to aim at.

        May return predicted positions during the grace period when the model
        temporarily loses detection of the current target.
        """
        if not enabled:
            self.reset()
            if boxes:
                return boxes[:1], confidences[:1]
            return [], []

        # No detections at all — try grace period
        if not boxes:
            return self._handle_no_detections()

        # Find the detection closest to crosshair
        aim_idx = self._find_closest_to_crosshair(boxes, crosshair_x, crosshair_y)
        if aim_idx < 0:
            return self._handle_no_detections()

        aim_box = boxes[aim_idx]
        aim_conf = confidences[aim_idx] if aim_idx < len(confidences) else 0.5

        # No current target — acquire
        if self._target is None:
            return self._acquire(aim_box, aim_conf)

        # Same target as before?
        if self._is_same_target(aim_box, self._target.box):
            self._update_velocity(aim_box)
            self._lock_score = min(
                self.MAX_LOCK_SCORE, self._lock_score + self.LOCK_SCORE_GAIN
            )
            self._target.box = list(aim_box)
            self._target.confidence = aim_conf
            self._target.frames_without_detection = 0
            self._frames_without_match = 0
            return [list(aim_box)], [aim_conf]

        # Different target — apply hysteresis before switching
        self._frames_without_match += 1

        aim_cx = (aim_box[0] + aim_box[2]) * 0.5
        aim_cy = (aim_box[1] + aim_box[3]) * 0.5
        dist_sq = (aim_cx - crosshair_x) ** 2 + (aim_cy - crosshair_y) ** 2

        # Quick switch if very close to crosshair or enough frames passed
        if dist_sq < 400 or self._frames_without_match >= self.SWITCH_FRAMES:
            return self._acquire(aim_box, aim_conf)

        # Not ready to switch — return empty to avoid flicking
        return [], []

    def reset(self) -> None:
        """Reset all sticky aim state."""
        self._target = None
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._lock_score = 0.0
        self._frames_without_match = 0

    # ---- private helpers ---------------------------------------------------

    @staticmethod
    def _find_closest_to_crosshair(boxes: List[List[float]], cx: int, cy: int) -> int:
        best_idx = -1
        best_d2 = float("inf")
        for i, box in enumerate(boxes):
            bx = (box[0] + box[2]) * 0.5
            by = (box[1] + box[3]) * 0.5
            d2 = (bx - cx) ** 2 + (by - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        return best_idx

    def _is_same_target(self, new_box: List[float], old_box: List[float]) -> bool:
        new_cx = (new_box[0] + new_box[2]) * 0.5
        new_cy = (new_box[1] + new_box[3]) * 0.5
        old_cx = (old_box[0] + old_box[2]) * 0.5
        old_cy = (old_box[1] + old_box[3]) * 0.5

        dist_sq = (new_cx - old_cx) ** 2 + (new_cy - old_cy) ** 2

        old_area = max(1.0, (old_box[2] - old_box[0]) * (old_box[3] - old_box[1]))
        target_size = math.sqrt(old_area)
        tracking_radius = target_size * 3.0

        new_area = max(1.0, (new_box[2] - new_box[0]) * (new_box[3] - new_box[1]))
        size_ratio = min(old_area, new_area) / max(old_area, new_area)

        return dist_sq < tracking_radius**2 and size_ratio > self.SIZE_RATIO_THRESHOLD

    def _update_velocity(self, new_box: List[float]) -> None:
        if self._target is None:
            return

        old_box = self._target.box
        old_area = max(1.0, (old_box[2] - old_box[0]) * (old_box[3] - old_box[1]))

        # Size-adaptive smoothing: distant targets get more smoothing
        size_factor = max(1.0, min(3.0, self.REFERENCE_TARGET_SIZE / old_area))
        smoothing = min(0.9, 0.6 + size_factor * 0.1)
        new_weight = 1.0 - smoothing

        old_cx = (old_box[0] + old_box[2]) * 0.5
        old_cy = (old_box[1] + old_box[3]) * 0.5
        new_cx = (new_box[0] + new_box[2]) * 0.5
        new_cy = (new_box[1] + new_box[3]) * 0.5

        raw_vx = new_cx - old_cx
        raw_vy = new_cy - old_cy

        self._vel_x = self._vel_x * smoothing + raw_vx * new_weight
        self._vel_y = self._vel_y * smoothing + raw_vy * new_weight

    def _handle_no_detections(self) -> Tuple[List[List[float]], List[float]]:
        if self._target is None:
            self.reset()
            return [], []

        self._target.frames_without_detection += 1

        if self._target.frames_without_detection <= self.MAX_FRAMES_WITHOUT_TARGET:
            # Grace period: extrapolate using velocity
            old_box = self._target.box
            frames = self._target.frames_without_detection

            old_cx = (old_box[0] + old_box[2]) * 0.5
            old_cy = (old_box[1] + old_box[3]) * 0.5
            pred_cx = old_cx + self._vel_x * frames
            pred_cy = old_cy + self._vel_y * frames

            bw = old_box[2] - old_box[0]
            bh = old_box[3] - old_box[1]
            predicted = [
                pred_cx - bw * 0.5,
                pred_cy - bh * 0.5,
                pred_cx + bw * 0.5,
                pred_cy + bh * 0.5,
            ]

            decayed = max(0.1, self._target.confidence * (1.0 - frames * 0.2))
            self._lock_score *= self.LOCK_SCORE_DECAY
            return [predicted], [decayed]

        self.reset()
        return [], []

    def _acquire(
        self, box: List[float], conf: float
    ) -> Tuple[List[List[float]], List[float]]:
        self._target = _TrackedTarget(box=list(box), confidence=conf)
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._lock_score = self.LOCK_SCORE_GAIN
        self._frames_without_match = 0
        return [list(box)], [conf]


# ---------------------------------------------------------------------------
# EMA Mouse Output Smoother (from Aimmy V2 MouseManager.cs)
# ---------------------------------------------------------------------------


class EMASmoother:
    """Exponential Moving Average smoother for mouse movement output.

    Applied AFTER PID controller, BEFORE send_mouse_move.
    Reduces jerkiness in the final mouse movement.
    """

    def __init__(self, smoothing_factor: float = 0.35):
        self.alpha = smoothing_factor
        self._last_dx: float = 0.0
        self._last_dy: float = 0.0
        self._initialized: bool = False

    def smooth(self, dx: float, dy: float) -> Tuple[float, float]:
        """Apply EMA smoothing to mouse movement deltas."""
        if not self._initialized:
            self._last_dx = dx
            self._last_dy = dy
            self._initialized = True
            return dx, dy

        out_dx = self._last_dx * self.alpha + dx * (1.0 - self.alpha)
        out_dy = self._last_dy * self.alpha + dy * (1.0 - self.alpha)

        self._last_dx = out_dx
        self._last_dy = out_dy

        return out_dx, out_dy

    def reset(self) -> None:
        self._last_dx = 0.0
        self._last_dy = 0.0
        self._initialized = False
