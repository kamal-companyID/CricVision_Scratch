import math
from ball_tracking.logger import get_logger
from ball_tracking.config import (
    ANGLE_CHANGE_THRESHOLD, MIN_GAP_BETWEEN_CHANGES, PITCH_MATCH_TOLERANCE,
)

logger = get_logger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _angle_between(v1: tuple, v2: tuple) -> float:
    """Angle (degrees) between two 2-D vectors."""
    m1 = math.hypot(*v1)
    m2 = math.hypot(*v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (m1 * m2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))


# ── Core: find up to 2 angle-change points ──────────────────────────────────

def _find_angle_changes(
    points: list[tuple[int, int]],
    threshold: float = ANGLE_CHANGE_THRESHOLD,
    min_gap: float = MIN_GAP_BETWEEN_CHANGES,
) -> list[tuple[int, int]]:
    """Walk consecutive triplets and collect up to 2 points where the
    trajectory angle changes by more than *threshold* degrees.

    A minimum pixel gap between the two points prevents frame-to-frame
    noise from producing a false second detection.
    """
    changes: list[tuple[int, int]] = []

    for i in range(1, len(points) - 1):
        prev, curr, nxt = points[i - 1], points[i], points[i + 1]

        v1 = (curr[0] - prev[0], curr[1] - prev[1])
        v2 = (nxt[0]  - curr[0], nxt[1]  - curr[1])

        angle = _angle_between(v1, v2)

        logger.debug("triplet: prev=%s, curr=%s, nxt=%s, angle=%.2f°", prev, curr, nxt, angle)
        if angle >= threshold:
            # First change — always accept
            if not changes:
                changes.append(curr)
            # Second change — only if far enough from the first
            elif _distance(curr, changes[0]) >= min_gap:
                changes.append(curr)
                break                     # we only need 2

    return changes


# ── Public API ───────────────────────────────────────────────────────────────

def find_impact_point(
    ball_path_points:   list[tuple[int, int]],
    pitch_point:        tuple[int, int] | None,
    ball_in_bat_points: list[tuple[int, int]],
) -> tuple[int, int] | bool:
    """Locate the ball–bat impact point using angle-change analysis.

    Logic (ported from analyzer.py):
        1. Find up to 2 major angle changes in the trajectory.
        2. Use the pitch presence + number of angle changes to classify:

           ┌─────────────────────────┬──────────────────────────────────────┐
           │ Situation               │ Impact point                         │
           ├─────────────────────────┼──────────────────────────────────────┤
           │ pitch + 2 angle changes │ 2nd change (with full-toss guard)    │
           │ pitch + 1 angle change  │ that change (if far enough from      │
           │                         │ pitch, else no impact)               │
           │ pitch + 0 angle changes │ no impact detected                   │
           │ no pitch + any changes  │ 1st change (full-toss / direct hit)  │
           │ no pitch + 0 changes    │ last bat-bbox point (fallback)       │
           └─────────────────────────┴──────────────────────────────────────┘

    Returns the impact point, or False if nothing could be determined.
    """
    if not ball_path_points:
        return False

    # Treat False (from pitch_point.py) the same as None
    if not pitch_point:
        pitch_point = None

    angle_changes = _find_angle_changes(ball_path_points)
    num_changes   = len(angle_changes)
    bat_pt        = ball_in_bat_points[-1] if ball_in_bat_points else None

    # ── PITCHED delivery ────────────────────────────────────────────────
    if pitch_point:

        if num_changes == 2:
            first_change  = angle_changes[0]
            second_change = angle_changes[1]

            # Full-toss guard: if the pitch point is actually just the
            # bat-impact Y-peak (pitch ≈ 2nd angle change), it's a false
            # pitch — treat as full-toss.  Impact = that 2nd change.
            if _distance(pitch_point, second_change) <= PITCH_MATCH_TOLERANCE:
                logger.info("[full-toss guard] pitch %s ≈ 2nd angle change %s — treating as full-toss",
                            pitch_point, second_change)
                return second_change

            logger.info("[pitch + 2 changes] first_change=%s, second_change=%s", first_change, second_change)
            logger.debug("distance pitch→1st=%.1f, pitch→2nd=%.1f",
                         _distance(pitch_point, first_change), _distance(pitch_point, second_change))
            # Standard bounce: 1st change ≈ pitch, 2nd change = impact
            return second_change

        if num_changes == 1:
            only_change = angle_changes[0]

            # If the single change IS the pitch, there's no bat impact
            if _distance(pitch_point, only_change) <= PITCH_MATCH_TOLERANCE:
                logger.info("[pitch only] single angle change matches pitch — no bat impact")
                return bat_pt or False

            # Otherwise, the change is the bat impact
            logger.info("[pitch + 1 change] only_change=%s", only_change)
            logger.debug("distance pitch→change=%.1f", _distance(pitch_point, only_change))
            return only_change

        # 0 angle changes — no bat contact detected
        return bat_pt or False

    # ── FULL-TOSS / no pitch ────────────────────────────────────────────
    if num_changes >= 1:
        logger.info("[full-toss] num_changes=%d, angle_changes=%s", num_changes, angle_changes)
        return angle_changes[0]

    # Absolute fallback: use the last ball-inside-bat-bbox point
    logger.info("[no pitch] num_changes=%d, bat_pt=%s", num_changes, bat_pt)
    return bat_pt or False
