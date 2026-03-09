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

# ── Adaptive-scale config ─────────────────────────────────────────────────────
DRAW_REF_SIZE  = 720        # reference dimension (px) — scale = 1.0 at this
DRAW_SCALE_MIN = 0.4
DRAW_SCALE_MAX = 2.5

# ── Gradient endpoints (BGR) ──────────────────────────────────────────────────
# Delivery: cyan → yellow       Bounce: yellow → magenta
GRAD_START = (255, 255,   0)    # cyan
GRAD_MID   = (0,   255, 255)    # yellow  (shared junction at pitch point)
GRAD_END   = (255,   0, 255)    # magenta

# ── Key-point colours (BGR) ───────────────────────────────────────────────────
FIRST_POINT_COLOR  = (255, 80,   0)   # blue
PITCH_POINT_COLOR  = (0,  140, 255)   # orange
IMPACT_POINT_COLOR = (0,    0, 255)   # red

# ── Path alpha (transparency) ────────────────────────────────────────────────
PATH_ALPHA = 0.85


# ── Scale helper ──────────────────────────────────────────────────────────────

def _get_draw_scale(frame: np.ndarray) -> float:
    """Return a multiplier that keeps drawing sizes proportional to resolution."""
    h, w = frame.shape[:2]
    return max(DRAW_SCALE_MIN, min(min(h, w) / float(DRAW_REF_SIZE), DRAW_SCALE_MAX))


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
    color_start: tuple[int, int, int],
    color_end:   tuple[int, int, int],
    alpha: float = PATH_ALPHA,
    num_segments: int | None = None,
) -> np.ndarray:
    """Draw a gradient path with alpha blending — returns the blended frame.

    Thickness adapts automatically to the frame resolution.
    """
    if not points or len(points) < 2:
        return frame

    scale = _get_draw_scale(frame)
    thickness = max(2, int(9 * scale))
    overlay = frame.copy()

    seg_count = min(num_segments, len(points)) if num_segments is not None else len(points)
    divisor = max(1, seg_count - 1)

    for i in range(1, seg_count):
        t = i / divisor
        color = _lerp_color(color_start, color_end, t)
        cv2.line(overlay, points[i - 1], points[i], color, thickness, cv2.LINE_AA)

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
    radius = max(6, int(14 * scale))
    border = max(1, int(2 * scale))

    cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, point, radius, (255, 255, 255), border, cv2.LINE_AA)

    if label:
        font_scale = 0.55 * scale
        font_thick = max(1, int(2 * scale))
        text_x = point[0] + radius + int(5 * scale)
        text_y = point[1] + int(5 * scale)
        # Shadow for readability
        cv2.putText(frame, label,
                    (text_x + 1, text_y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), font_thick + 1, cv2.LINE_AA)
        cv2.putText(frame, label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color, font_thick, cv2.LINE_AA)


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
    delivery = path_data.get('delivery')
    bounce   = path_data.get('bounce')

    if delivery:
        frame = _draw_gradient_path(frame, delivery, GRAD_START, GRAD_MID)
    if bounce:
        frame = _draw_gradient_path(frame, bounce, GRAD_MID, GRAD_END)

    if first_point:
        _draw_event_point(frame, first_point, FIRST_POINT_COLOR,
                          'First' if show_labels else '')
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

    delivery = path_data.get('delivery')
    bounce   = path_data.get('bounce')

    # ── Delivery arc (progress 0.0 → 0.5) ───────────────────────────────────
    if delivery:
        d_progress = min(1.0, progress / 0.5)
        n = max(2, int(len(delivery) * d_progress))
        frame = _draw_gradient_path(frame, delivery[:n], GRAD_START, GRAD_MID, num_segments=n)

    # ── Bounce arc (progress 0.5 → 1.0) ─────────────────────────────────────
    if bounce and progress > 0.5:
        b_progress = min(1.0, (progress - 0.5) / 0.5)
        n = max(2, int(len(bounce) * b_progress))
        frame = _draw_gradient_path(frame, bounce[:n], GRAD_MID, GRAD_END, num_segments=n)

    # ── Key points (appear when their arc reaches them) ──────────────────────
    if first_point and progress > 0.0:
        _draw_event_point(frame, first_point, FIRST_POINT_COLOR,
                          'First' if show_labels else '')
    if pitch_point and progress >= 0.5:
        _draw_event_point(frame, pitch_point, PITCH_POINT_COLOR,
                          'Pitch' if show_labels else '')
    if impact_point and progress >= 1.0:
        _draw_event_point(frame, impact_point, IMPACT_POINT_COLOR,
                          'Impact' if show_labels else '')
    return frame
