from __future__ import annotations

from dataclasses import dataclass, field

from .smart_tracker import SmartTracker


@dataclass
class LoopState:
    last_pid_update: float = 0.0
    last_ddxoft_stats_time: float = 0.0

    last_method_check_time: float = 0.0
    cached_mouse_move_method: str = "mouse_event"

    pid_check_interval: float = 1.0
    ddxoft_stats_interval: float = 30.0
    method_check_interval: float = 2.0

    bezier_curve_scalar: float = 0.0
    target_locked: bool = False

    smart_tracker: SmartTracker | None = None
    tracker_last_time: float = 0.0
    tracker_last_target_box: tuple | None = None

    aiming_start_time: float = 0.0

    target_confirm_count: int = 0
    target_last_box: tuple | None = None
    target_last_seen_time: float = 0.0
    smoothed_box: list | None = None
