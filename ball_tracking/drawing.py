"""
drawing.py — all visual overlay helpers for ball-path analysis.

Includes:
  • Adaptive scaling for different resolutions
  • Gradient-coloured path rendering with alpha blending
  • Event-point markers (first / pitch / impact)
  • Animated and static draw functions
"""

import cv2
import numpy as np
from ball_tracking.config import (
    DRAW_REF_SIZE, DRAW_SCALE_MIN, DRAW_SCALE_MAX,
    GRAD_START, GRAD_END,
    PITCH_POINT_COLOR, IMPACT_POINT_COLOR,
    PATH_ALPHA,
)


# ── Scale helper ──────────────────────────────────────────────────────────────

def _get_draw_scale(frame: np.ndarray) -> float:
    """Return a multiplier that keeps drawing sizes proportional to resolution."""
    h, w = frame.shape[:2]
    return max(DRAW_SCALE_MIN, min(min(h, w) / float(DRAW_REF_SIZE), DRAW_SCALE_MAX))


def _merge_path_segments(path_data: dict) -> list[tuple[int, int]]:
    """Return a continuous path from the delivery and bounce segments."""
    delivery = path_data.get('delivery') or []
    bounce = path_data.get('bounce') or []

    if delivery and bounce:
        if delivery[-1] == bounce[0]:
            return [*delivery, *bounce[1:]]
        return [*delivery, *bounce]
    if delivery:
        return delivery
    if bounce:
        return bounce
    return []


# ── Colour interpolation ─────────────────────────────────────────────────────

def _lerp_color(
    c1: tuple[int, int, int],
    c2: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    """Linearly interpolate between two BGR colours (t in 0→1)."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


# ── Low-level primitives ──────────────────────────────────────────────────────

def _draw_gradient_path(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    alpha: float = PATH_ALPHA,
    num_segments: int | None = None,
) -> np.ndarray:
    """Draw a gradient path with alpha blending — returns the blended frame.

    Thickness adapts automatically to the frame resolution.
    """
    if not points or len(points) < 2:
        return frame

    scale = _get_draw_scale(frame)
    thickness = max(2, int(6 * scale))
    overlay = frame.copy()

    seg_count = min(num_segments, len(points)) if num_segments is not None else len(points)

    for i in range(1, seg_count):
        t = i / seg_count
        color = _lerp_color(GRAD_START, GRAD_END, t)
        cv2.line(overlay, points[i - 1], points[i], color, thickness)

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def _draw_event_point(
    frame: np.ndarray,
    point: tuple[int, int],
    color: tuple[int, int, int],
    label: str = '',
) -> None:
    """Draw a filled circle with a white border and optional label at *point*."""
    if not point:
        return

    scale = _get_draw_scale(frame)
    radius = max(4, int(7 * scale))
    border = max(1, int(1 * scale))

    cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, point, radius, (255, 255, 255), border, cv2.LINE_AA)

    # if label:
    #     font_scale = 0.55 * scale
    #     font_thick = max(1, int(2 * scale))
    #     text_x = point[0] + radius + int(5 * scale)
    #     text_y = point[1] + int(5 * scale)
    #     # Shadow for readability
    #     cv2.putText(frame, label,
    #                 (text_x + 1, text_y + 1),
    #                 cv2.FONT_HERSHEY_SIMPLEX, font_scale,
    #                 (0, 0, 0), font_thick + 1, cv2.LINE_AA)
    #     cv2.putText(frame, label,
    #                 (text_x, text_y),
    #                 cv2.FONT_HERSHEY_SIMPLEX, font_scale,
    #                 color, font_thick, cv2.LINE_AA)


# ── High-level composite draws ────────────────────────────────────────────────

def draw_ball_path(
    frame,
    path_data: dict,
    first_point:  tuple[int, int] | None = None,
    pitch_point:  tuple[int, int] | None = None,
    impact_point: tuple[int, int] | None = None,
    show_labels: bool = True,
):
    """Draw the complete ball path (gradient arcs + key points).

    Returns the blended frame (alpha compositing means the frame object
    may be replaced — always use the return value).
    """
    full_path = _merge_path_segments(path_data)

    if full_path:
        frame = _draw_gradient_path(frame, full_path)

    # if first_point:
    #     _draw_event_point(frame, first_point, FIRST_POINT_COLOR,
    #                       'First' if show_labels else '')
    if pitch_point:
        _draw_event_point(frame, pitch_point, PITCH_POINT_COLOR,
                          'Pitch' if show_labels else '')
    if impact_point:
        _draw_event_point(frame, impact_point, IMPACT_POINT_COLOR,
                          'Impact' if show_labels else '')
    return frame


def draw_ball_path_animated(
    frame,
    path_data: dict,
    progress: float,
    first_point:  tuple[int, int] | None = None,
    pitch_point:  tuple[int, int] | None = None,
    impact_point: tuple[int, int] | None = None,
    show_labels: bool = True,
):
    """Draw a partially-revealed ball path controlled by *progress* (0 → 1).

    0.0 → nothing    0.5 → delivery complete    1.0 → everything complete.
    Returns the blended frame.
    """
    progress = max(0.0, min(1.0, progress))

    delivery = path_data.get('delivery') or []
    bounce = path_data.get('bounce') or []
    visible_points: list[tuple[int, int]] = []

    # ── Delivery arc (progress 0.0 → 0.5) ───────────────────────────────────
    if delivery:
        d_progress = min(1.0, progress / 0.5)
        n = max(2, int(len(delivery) * d_progress))
        visible_points = delivery[:n]

    # ── Bounce arc (progress 0.5 → 1.0) ─────────────────────────────────────
    if bounce and progress > 0.5:
        b_progress = min(1.0, (progress - 0.5) / 0.5)
        n = max(2, int(len(bounce) * b_progress))
        visible_bounce = bounce[:n]
        if visible_points and visible_bounce:
            if visible_points[-1] == visible_bounce[0]:
                visible_points = [*visible_points, *visible_bounce[1:]]
            else:
                visible_points = [*visible_points, *visible_bounce]
        else:
            visible_points = visible_bounce

    if visible_points:
        frame = _draw_gradient_path(frame, visible_points)

    # ── Key points (appear when their arc reaches them) ──────────────────────
    # if first_point and progress > 0.0:
    #     _draw_event_point(frame, first_point, FIRST_POINT_COLOR,
    #                       'First' if show_labels else '')
    if pitch_point and progress >= 0.5:
        _draw_event_point(frame, pitch_point, PITCH_POINT_COLOR,
                          'Pitch' if show_labels else '')
    if impact_point and progress >= 1.0:
        _draw_event_point(frame, impact_point, IMPACT_POINT_COLOR,
                          'Impact' if show_labels else '')
    return frame
