"""Main loop for AI inference and mouse control"""

from __future__ import annotations

import ctypes
import os
import queue
import threading
import time
import traceback
from typing import TYPE_CHECKING, List, Tuple

import cv2
import numpy as np

from win_utils import is_key_pressed

from .ai_aiming import process_aiming
from .ai_loop_state import LoopState
from .ai_loop_utils import (
    apply_temporal_filter,
    calculate_detection_region,
    clear_queues,
    filter_boxes_by_fov,
    find_closest_target,
    update_crosshair_position,
    update_queues,
)
from .inference import (
    PIDController,
    non_max_suppression,
    postprocess_outputs,
    preprocess_image,
    preprocess_image_letterbox,
)
from .screen_capture import (
    _cleanup_capture,
    capture_frame,
    initialize_screen_capture,
    reinitialize_if_method_changed,
)
from .session_utils import create_run_options

if TYPE_CHECKING:
    import onnxruntime as ort

    from .config import Config


def _run_tta(
    frame: np.ndarray,
    model: ort.InferenceSession,
    input_name: str,
    region: dict,
    model_input_size: int,
    min_confidence: float,
    lb_scale: float,
    lb_padding: tuple | None,
    run_options: object | None = None,
) -> Tuple[List[List[float]], List[float]]:
    original_blob = preprocess_image(frame, model_input_size)

    outputs = model.run(None, {input_name: original_blob}, run_options=run_options)
    boxes, confidences = postprocess_outputs(
        outputs,
        region["width"],
        region["height"],
        model_input_size,
        min_confidence,
        region["left"],
        region["top"],
        letterbox_scale=lb_scale,
        letterbox_padding=lb_padding,
    )

    flipped = frame[:, ::-1].copy()
    flipped_blob = preprocess_image(flipped, model_input_size)
    outputs_flip = model.run(None, {input_name: flipped_blob}, run_options=run_options)
    boxes_flip, conf_flip = postprocess_outputs(
        outputs_flip,
        region["width"],
        region["height"],
        model_input_size,
        min_confidence,
        region["left"],
        region["top"],
        letterbox_scale=lb_scale,
        letterbox_padding=lb_padding,
    )

    if boxes_flip:
        frame_w = region["width"]
        for i, box in enumerate(boxes_flip):
            x1, y1, x2, y2 = box
            boxes.append(
                [
                    frame_w - x2 + 2 * region["left"],
                    y1,
                    frame_w - x1 + 2 * region["left"],
                    y2,
                ]
            )
            confidences.append(conf_flip[i])

    if boxes:
        boxes, confidences = non_max_suppression(boxes, confidences)

    return boxes, confidences


def _run_multi_scale(
    frame: np.ndarray,
    model: ort.InferenceSession,
    input_name: str,
    region: dict,
    model_input_size: int,
    min_confidence: float,
    run_options: object | None = None,
) -> Tuple[List[List[float]], List[float]]:
    all_boxes: List[List[float]] = []
    all_confs: List[float] = []

    half_w = region["width"] // 2
    half_h = region["height"] // 2
    sub_regions = [
        (0, 0, half_w, half_h),
        (half_w, 0, region["width"], half_h),
        (0, half_h, half_w, region["height"]),
        (half_w, half_h, region["width"], region["height"]),
    ]

    frame_h, frame_w = frame.shape[:2]

    for sx1, sy1, sx2, sy2 in sub_regions:
        abs_x1 = int(sx1)
        abs_y1 = int(sy1)
        abs_x2 = min(int(sx2), frame_w)
        abs_y2 = min(int(sy2), frame_h)
        if abs_x2 - abs_x1 < 100 or abs_y2 - abs_y1 < 100:
            continue

        crop = frame[abs_y1:abs_y2, abs_x1:abs_x2]
        if crop.size == 0:
            continue

        crop_input = preprocess_image(crop, model_input_size)
        crop_outputs = model.run(
            None, {input_name: crop_input}, run_options=run_options
        )
        crop_boxes, crop_confs = postprocess_outputs(
            crop_outputs,
            abs_x2 - abs_x1,
            abs_y2 - abs_y1,
            model_input_size,
            min_confidence * 0.7,
            offset_x=region["left"] + abs_x1,
            offset_y=region["top"] + abs_y1,
            min_box_area_ratio=0.00002,
        )

        all_boxes.extend(crop_boxes)
        all_confs.extend(crop_confs)

    if all_boxes:
        all_boxes, all_confs = non_max_suppression(all_boxes, all_confs)

    return all_boxes, all_confs


def _run_pass2_refinement(
    frame: np.ndarray,
    boxes: List[List[float]],
    confidences: List[float],
    model: ort.InferenceSession,
    input_name: str,
    region: dict,
    model_input_size: int,
    min_confidence: float,
    min_refine_area: float = 2000.0,
    run_options: object | None = None,
) -> Tuple[List[List[float]], List[float]]:
    if not boxes:
        return boxes, confidences

    frame_h, frame_w = frame.shape[:2]
    refined_boxes: List[List[float]] = []
    refined_confs: List[float] = []

    for box, conf in zip(boxes, confidences):
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        box_area = box_w * box_h

        if box_area >= min_refine_area:
            refined_boxes.append(box)
            refined_confs.append(conf)
            continue

        margin_x = max(box_w * 3, 80)
        margin_y = max(box_h * 3, 80)

        rel_x1 = box[0] - region["left"]
        rel_y1 = box[1] - region["top"]
        rel_x2 = box[2] - region["left"]
        rel_y2 = box[3] - region["top"]

        crop_x1 = max(0, int(rel_x1 - margin_x))
        crop_y1 = max(0, int(rel_y1 - margin_y))
        crop_x2 = min(frame_w, int(rel_x2 + margin_x))
        crop_y2 = min(frame_h, int(rel_y2 + margin_y))

        if crop_x2 - crop_x1 < 60 or crop_y2 - crop_y1 < 60:
            refined_boxes.append(box)
            refined_confs.append(conf)
            continue

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_input = preprocess_image(crop, model_input_size)
        crop_outputs = model.run(
            None, {input_name: crop_input}, run_options=run_options
        )
        crop_boxes, crop_confs = postprocess_outputs(
            crop_outputs,
            crop_x2 - crop_x1,
            crop_y2 - crop_y1,
            model_input_size,
            min_confidence * 0.6,
            offset_x=region["left"] + crop_x1,
            offset_y=region["top"] + crop_y1,
            min_box_area_ratio=0.00002,
        )

        if crop_boxes:
            best_idx = max(range(len(crop_confs)), key=lambda i: crop_confs[i])
            refined_boxes.append(crop_boxes[best_idx])
            refined_confs.append(crop_confs[best_idx])
        else:
            refined_boxes.append(box)
            refined_confs.append(conf)

    if len(refined_boxes) > 1:
        refined_boxes, refined_confs = non_max_suppression(refined_boxes, refined_confs)

    return refined_boxes, refined_confs


def _try_hot_swap_model(
    config: Config, model: ort.InferenceSession, current_model_path: str
):
    """Try hot-swapping ONNX model when config.model_path changes."""

    if config.model_path == current_model_path:
        return model, current_model_path, model.get_inputs()[0].name

    new_model_path = config.model_path
    if not os.path.isabs(new_model_path):
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        abs_model_path = os.path.join(project_root, new_model_path)
    else:
        abs_model_path = new_model_path

    if not (os.path.exists(abs_model_path) and abs_model_path.endswith(".onnx")):
        print(f"[模型熱切換] 路徑無效或檔案不存在: {abs_model_path}")
        config.model_path = current_model_path
        return model, current_model_path, model.get_inputs()[0].name

    try:
        from .session_utils import create_inference_session

        new_model = create_inference_session(abs_model_path, config)

        input_name = new_model.get_inputs()[0].name
        actual_providers = new_model.get_providers()
        if actual_providers:
            config.current_provider = actual_providers[0]
        print(f"[模型熱切換] 已切換至: {os.path.basename(abs_model_path)}")
        return new_model, new_model_path, input_name
    except Exception as e:
        print(f"[模型熱切換] 載入失敗: {e}，繼續使用原模型")
        config.model_path = current_model_path
        return model, current_model_path, model.get_inputs()[0].name


def _sleep_precise(seconds: float) -> None:
    """Sleep with better precision for very short intervals on Windows."""

    if seconds <= 0:
        return

    if seconds >= 0.002:
        time.sleep(seconds)
        return

    # Reduce CPU spin on sub-2ms waits:
    # 1) cooperatively yield while remaining time is still relatively large
    # 2) only busy-wait in a very small tail window for precision
    deadline = time.perf_counter() + seconds
    spin_threshold = 0.0002  # 0.2ms spin window

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break

        if remaining > spin_threshold:
            # Yield the timeslice to avoid burning a full CPU core.
            # Keep a tiny safety margin so final timing still relies on perf_counter.
            sleep_for = max(0.0, remaining - spin_threshold)
            if sleep_for >= 0.001:
                time.sleep(sleep_for)
            else:
                time.sleep(0)


def _set_windows_timer_resolution_1ms(enable: bool) -> bool:
    """Enable/disable 1ms timer resolution on Windows. Returns success status."""

    if os.name != "nt":
        return False

    try:
        winmm = ctypes.WinDLL("winmm")
        if enable:
            return winmm.timeBeginPeriod(1) == 0
        return winmm.timeEndPeriod(1) == 0
    except Exception:
        return False


def ai_logic_loop(
    config: Config,
    model: ort.InferenceSession,
    model_type: str,
    overlay_boxes_queue: queue.Queue,
    overlay_confidences_queue: queue.Queue,
    auto_fire_boxes_queue: queue.Queue | None = None,
) -> None:
    """AI 推理和滑鼠控制的主要循環"""

    input_name = model.get_inputs()[0].name
    run_options = create_run_options()

    pid_x = PIDController(config.pid_kp_x, config.pid_ki_x, config.pid_kd_x)
    pid_y = PIDController(config.pid_kp_y, config.pid_ki_y, config.pid_kd_y)

    state = LoopState(cached_mouse_move_method=config.mouse_move_method)
    current_model_path = config.model_path

    ema_total = 0.0
    ema_capture = 0.0
    ema_pre = 0.0
    ema_inf = 0.0
    ema_post = 0.0
    last_stats_print = time.perf_counter()
    last_detection_run_time = 0.0

    capture_lock = threading.Lock()
    capture_stop_event = threading.Event()
    capture_state: dict[str, object] = {
        "latest_frame": None,
        "latest_region": None,
        "target_region": None,
    }
    _last_valid_frame: list = [None]  # mutable container for closure
    # Mutable containers so the capture worker can hot-swap the backend
    _capture_backend: list = [None]
    _active_method: list = [None]

    def _capture_worker() -> None:
        _capture_backend[0] = initialize_screen_capture(config)
        _active_method[0] = getattr(config, "screenshot_method", "mss")

        high_res_timer_enabled = False
        last_capture_perf = 0.0
        last_method_check = 0.0

        try:
            while config.Running and not capture_stop_event.is_set():
                screenshot_interval = max(
                    0.001,
                    float(
                        getattr(config, "screenshot_interval", config.detect_interval)
                    ),
                )
                should_use_high_res_timer = screenshot_interval <= 0.002

                if should_use_high_res_timer and not high_res_timer_enabled:
                    high_res_timer_enabled = _set_windows_timer_resolution_1ms(True)
                elif high_res_timer_enabled and not should_use_high_res_timer:
                    _set_windows_timer_resolution_1ms(False)
                    high_res_timer_enabled = False

                # --- Hot-swap screenshot backend every 0.5s ---
                now_check = time.perf_counter()
                if now_check - last_method_check >= 0.5:
                    last_method_check = now_check
                    new_backend, new_method = reinitialize_if_method_changed(
                        config,
                        _capture_backend[0],
                        _active_method[0],
                    )
                    if new_backend is not _capture_backend[0]:
                        _capture_backend[0] = new_backend
                        _active_method[0] = new_method
                        _last_valid_frame[0] = (
                            None  # reset cached frame on backend change
                        )

                with capture_lock:
                    target_region = capture_state.get("target_region")

                if target_region is None:
                    _sleep_precise(0.001)
                    continue

                now_capture = time.perf_counter()
                wait_for = screenshot_interval - (now_capture - last_capture_perf)
                if wait_for > 0:
                    _sleep_precise(wait_for)
                    continue

                last_capture_perf = time.perf_counter()
                captured_frame = capture_frame(_capture_backend[0], target_region)

                if captured_frame is not None:
                    _last_valid_frame[0] = captured_frame
                elif _last_valid_frame[0] is not None:
                    # dxcam returns None when screen content hasn't changed;
                    # reuse the last valid frame so FPS isn't throttled by VSync
                    captured_frame = _last_valid_frame[0]
                else:
                    continue

                with capture_lock:
                    capture_state["latest_frame"] = captured_frame
                    capture_state["latest_region"] = target_region

                config.last_screenshot_time = time.time()
                config.screenshot_frame_count = (
                    int(getattr(config, "screenshot_frame_count", 0)) + 1
                )
        finally:
            if high_res_timer_enabled:
                _set_windows_timer_resolution_1ms(False)
            if _capture_backend[0] is not None:
                _cleanup_capture(_capture_backend[0])

    capture_thread = threading.Thread(
        target=_capture_worker, name="CaptureWorker", daemon=True
    )
    capture_thread.start()

    try:
        while config.Running:
            try:
                loop_start = time.perf_counter()
                current_time = time.time()

                model, current_model_path, input_name = _try_hot_swap_model(
                    config, model, current_model_path
                )

                if current_time - state.last_pid_update > state.pid_check_interval:
                    pid_x.Kp, pid_x.Ki, pid_x.Kd = (
                        config.pid_kp_x,
                        config.pid_ki_x,
                        config.pid_kd_x,
                    )
                    pid_y.Kp, pid_y.Ki, pid_y.Kd = (
                        config.pid_kp_y,
                        config.pid_ki_y,
                        config.pid_kd_y,
                    )
                    state.last_pid_update = current_time

                if (
                    current_time - state.last_method_check_time
                    > state.method_check_interval
                ):
                    new_method = config.mouse_move_method
                    if new_method != state.cached_mouse_move_method:
                        state.cached_mouse_move_method = new_method
                    state.last_method_check_time = current_time

                update_crosshair_position(config, config.width // 2, config.height // 2)

                is_aiming = bool(getattr(config, "always_aim", False)) or any(
                    is_key_pressed(k) for k in config.AimKeys
                )
                if is_aiming:
                    if state.aiming_start_time == 0.0:
                        state.aiming_start_time = current_time
                else:
                    state.aiming_start_time = 0.0

                if not config.AimToggle or (
                    not config.keep_detecting and not is_aiming
                ):
                    clear_queues(overlay_boxes_queue, overlay_confidences_queue)
                    config.tracker_has_prediction = False
                    time.sleep(0.05)
                    continue

                crosshair_x, crosshair_y = config.crosshairX, config.crosshairY
                region = calculate_detection_region(config, crosshair_x, crosshair_y)
                if region["width"] <= 0 or region["height"] <= 0:
                    continue

                with capture_lock:
                    capture_state["target_region"] = region
                    latest_frame = capture_state.get("latest_frame")
                    latest_region = capture_state.get("latest_region")

                if latest_frame is None or latest_region is None:
                    _sleep_precise(0.001)
                    continue

                idle_enabled = getattr(config, "idle_detect_enabled", True)
                if getattr(config, "always_aim", False) or getattr(
                    config, "always_auto_fire", False
                ):
                    idle_enabled = False
                if idle_enabled and not is_aiming:
                    desired_interval = getattr(
                        config, "idle_detect_interval", config.detect_interval
                    )
                else:
                    desired_interval = config.detect_interval

                now_detect = time.perf_counter()
                elapsed_detect = now_detect - last_detection_run_time
                if elapsed_detect < desired_interval:
                    next_detect_wait = max(0.0, desired_interval - elapsed_detect)
                    if next_detect_wait > 0:
                        _sleep_precise(next_detect_wait)
                    continue
                last_detection_run_time = now_detect

                t0 = time.perf_counter()
                use_letterbox = getattr(config, "use_letterbox_preprocess", False)
                if use_letterbox:
                    input_tensor, lb_scale, lb_padding = preprocess_image_letterbox(
                        latest_frame, config.model_input_size
                    )
                else:
                    input_tensor = preprocess_image(
                        latest_frame, config.model_input_size
                    )
                    lb_scale, lb_padding = 0.0, None
                t1 = time.perf_counter()
                t2 = t3 = t4 = None

                try:
                    t2 = time.perf_counter()
                    outputs = model.run(
                        None, {input_name: input_tensor}, run_options=run_options
                    )
                    t3 = time.perf_counter()
                    boxes, confidences = postprocess_outputs(
                        outputs,
                        latest_region["width"],
                        latest_region["height"],
                        config.model_input_size,
                        config.min_confidence,
                        latest_region["left"],
                        latest_region["top"],
                        letterbox_scale=lb_scale,
                        letterbox_padding=lb_padding,
                    )

                    use_multi_scale = getattr(config, "multi_scale_inference", True)
                    region_min_dim = min(
                        latest_region["width"], latest_region["height"]
                    )
                    do_multi_scale = (
                        use_multi_scale and len(boxes) == 0 and region_min_dim > 320
                    )

                    if not do_multi_scale:
                        boxes, confidences = non_max_suppression(boxes, confidences)
                    else:
                        half_w = latest_region["width"] // 2
                        half_h = latest_region["height"] // 2
                        sub_regions = [
                            (0, 0, half_w, half_h),
                            (half_w, 0, latest_region["width"], half_h),
                            (0, half_h, half_w, latest_region["height"]),
                            (
                                half_w,
                                half_h,
                                latest_region["width"],
                                latest_region["height"],
                            ),
                        ]

                        frame_h, frame_w = latest_frame.shape[:2]

                        for sx1, sy1, sx2, sy2 in sub_regions:
                            abs_x1 = int(sx1)
                            abs_y1 = int(sy1)
                            abs_x2 = min(int(sx2), frame_w)
                            abs_y2 = min(int(sy2), frame_h)
                            if abs_x2 - abs_x1 < 100 or abs_y2 - abs_y1 < 100:
                                continue

                            crop = latest_frame[abs_y1:abs_y2, abs_x1:abs_x2]
                            if crop.size == 0:
                                continue

                            crop_input = preprocess_image(crop, config.model_input_size)
                            crop_outputs = model.run(
                                None, {input_name: crop_input}, run_options=run_options
                            )
                            crop_boxes, crop_confs = postprocess_outputs(
                                crop_outputs,
                                abs_x2 - abs_x1,
                                abs_y2 - abs_y1,
                                config.model_input_size,
                                config.min_confidence * 0.7,
                                offset_x=latest_region["left"] + abs_x1,
                                offset_y=latest_region["top"] + abs_y1,
                                min_box_area_ratio=0.00002,
                            )

                            boxes.extend(crop_boxes)
                            confidences.extend(crop_confs)

                        if boxes:
                            boxes, confidences = non_max_suppression(boxes, confidences)

                    t4 = time.perf_counter()
                    config.last_detection_time = time.time()
                    config.detection_frame_count = (
                        int(getattr(config, "detection_frame_count", 0)) + 1
                    )
                except (RuntimeError, ValueError) as e:
                    print(f"ONNX 推理錯誤: {e}")
                    continue

                boxes, confidences = filter_boxes_by_fov(
                    boxes, confidences, crosshair_x, crosshair_y, config.fov_size
                )

                if config.single_target_mode:
                    boxes, confidences = find_closest_target(
                        boxes,
                        confidences,
                        crosshair_x,
                        crosshair_y,
                        aim_part=config.aim_part,
                        head_height_ratio=config.head_height_ratio,
                    )

                boxes, confidences = apply_temporal_filter(
                    boxes,
                    confidences,
                    state,
                    current_time,
                    confirm_frames=int(getattr(config, "temporal_confirm_frames", 2)),
                    expire_time=float(getattr(config, "temporal_expire_time", 0.4)),
                )

                if is_aiming and boxes:
                    process_aiming(
                        config,
                        boxes,
                        crosshair_x,
                        crosshair_y,
                        pid_x,
                        pid_y,
                        state.cached_mouse_move_method,
                        state,
                        current_time,
                    )
                else:
                    state.target_locked = False
                    config.tracker_has_prediction = False
                    pid_x.reset()
                    pid_y.reset()

                update_queues(
                    overlay_boxes_queue,
                    overlay_confidences_queue,
                    boxes,
                    confidences,
                    auto_fire_queue=auto_fire_boxes_queue,
                )

                if getattr(config, "enable_latency_stats", False):
                    alpha = float(getattr(config, "latency_stats_alpha", 0.2))
                    total_ms = (time.perf_counter() - loop_start) * 1000.0
                    cap_ms = (t0 - loop_start) * 1000.0
                    pre_ms = (t1 - t0) * 1000.0
                    inf_ms = (
                        (t3 - t2) * 1000.0 if t3 is not None and t2 is not None else 0.0
                    )
                    post_ms = (
                        (t4 - t3) * 1000.0 if t4 is not None and t3 is not None else 0.0
                    )

                    ema_total = ema_total * (1 - alpha) + total_ms * alpha
                    ema_capture = ema_capture * (1 - alpha) + cap_ms * alpha
                    ema_pre = ema_pre * (1 - alpha) + pre_ms * alpha
                    ema_inf = ema_inf * (1 - alpha) + inf_ms * alpha
                    ema_post = ema_post * (1 - alpha) + post_ms * alpha

                    now = time.perf_counter()
                    if now - last_stats_print >= float(
                        getattr(config, "latency_stats_interval", 1.0)
                    ):
                        print(
                            f"[Latency EMA] total={ema_total:.1f}ms "
                            f"cap={ema_capture:.1f}ms pre={ema_pre:.1f}ms "
                            f"inf={ema_inf:.1f}ms post={ema_post:.1f}ms "
                            f"interval={desired_interval * 1000:.0f}ms"
                        )
                        last_stats_print = now

            except Exception as e:
                print(f"[AI Loop Error] {e}")
                traceback.print_exc()
                time.sleep(1.0)
    finally:
        capture_stop_event.set()
        capture_thread.join(timeout=1.0)
