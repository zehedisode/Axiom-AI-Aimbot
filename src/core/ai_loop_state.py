from __future__ import annotations

from dataclasses import dataclass

from .smart_tracker import EMASmoother, KalmanTracker, SmartTracker, StickyAimManager


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

    # Legacy SmartTracker (superseded by Kalman)
    smart_tracker: SmartTracker | None = None
    tracker_last_time: float = 0.0
    tracker_last_target_box: tuple | None = None

    aiming_start_time: float = 0.0

    target_confirm_count: int = 0
    target_last_box: tuple | None = None
    target_last_seen_time: float = 0.0
    smoothed_box: list | None = None

    # ── Aimmy V2 integration state ──
    kalman_tracker: KalmanTracker | None = None
    sticky_aim: StickyAimManager | None = None
    ema_smoother: EMASmoother | None = None

    # Dynamic bezier scalar tracking
    bezier_needs_new_scalar: bool = True

    def ensure_aimmy_systems(self, config: object) -> None:
        """Lazily initialize Aimmy V2 subsystems based on config."""
        # Kalman tracker
        use_kalman = getattr(config, "use_kalman_tracker", True)
        if use_kalman and self.kalman_tracker is None:
            self.kalman_tracker = KalmanTracker(
                process_noise=float(getattr(config, "kalman_process_noise", 0.001)),
                measurement_noise=float(
                    getattr(config, "kalman_measurement_noise", 0.01)
                ),
            )
        elif not use_kalman and self.kalman_tracker is not None:
            self.kalman_tracker.reset()
            self.kalman_tracker = None

        # Sticky aim
        sticky_enabled = getattr(config, "sticky_aim_enabled", True)
        if sticky_enabled and self.sticky_aim is None:
            self.sticky_aim = StickyAimManager()
        elif not sticky_enabled and self.sticky_aim is not None:
            self.sticky_aim.reset()
            self.sticky_aim = None

        # EMA smoother
        ema_enabled = getattr(config, "ema_mouse_smoothing", True)
        if ema_enabled and self.ema_smoother is None:
            self.ema_smoother = EMASmoother(
                smoothing_factor=float(getattr(config, "ema_smoothing_factor", 0.35)),
            )
        elif not ema_enabled and self.ema_smoother is not None:
            self.ema_smoother.reset()
            self.ema_smoother = None
