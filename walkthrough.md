# Walkthrough: [impact_point.py](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py) Rewrite

## What Changed

Completely rewrote [impact_point.py](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py) — replaced the 4 independent sub-detectors with analyzer.py's single **2-angle-change model**.

```diff:impact_point.py
import math

# ── Thresholds ──────────────────────────────────────────────────────────────
CONSECUTIVE_ANGLE_THRESHOLD = 30   # degrees — sharp per-segment angle change
ANCHORED_ANGLE_THRESHOLD    = 45   # degrees — cumulative drift from post-pitch dir
MIN_SEGMENT_PX              = 4    # min vector length — filters sub-pixel jitter
MAX_SEGMENT_PX              = 150  # max vector length — filters false-detection jumps


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _angle_deg(v1: tuple, v2: tuple) -> float:
    m1, m2 = math.hypot(*v1), math.hypot(*v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    cos_a = (v1[0]*v2[0] + v1[1]*v2[1]) / (m1 * m2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))


# ── Sub-detectors ────────────────────────────────────────────────────────────

def _turning_point(
    points: list[tuple[int, int]],
    min_seg: int = MIN_SEGMENT_PX,
    max_seg: int = MAX_SEGMENT_PX,
) -> tuple[int, int] | None:
    """Return the first point where the Y-component of velocity reverses sign.

    On a pitched delivery the ball travels upward (y decreasing) toward the
    batsman; bat contact deflects it downward (y increasing).  The peak is the
    best single-frame proxy for the contact instant.

    Segments shorter than *min_seg* (noise) or longer than *max_seg* (false
    detections / large jumps) are both skipped.
    """
    for i in range(1, len(points) - 1):
        seg = math.hypot(points[i][0]-points[i-1][0], points[i][1]-points[i-1][1])
        # if seg < min_seg or seg > max_seg:
        #     continue
        dy_prev = points[i][1]   - points[i-1][1]
        dy_next = points[i+1][1] - points[i][1]
        # Strict sign reversal (== 0 ignored — avoids flat-arc false positives)
        if (dy_prev < 0 and dy_next > 0) or (dy_prev > 0 and dy_next < 0):
            print(f"Y-reversal at {points[i]}  (dy_prev={dy_prev}, dy_next={dy_next})")
            return points[i]
    return None


def _consecutive_change(
    points: list[tuple[int, int]],
    threshold: float = CONSECUTIVE_ANGLE_THRESHOLD,
    min_seg: int = MIN_SEGMENT_PX,
    max_seg: int = MAX_SEGMENT_PX,
) -> tuple[int, int] | None:
    """First point where the angle between consecutive movement vectors exceeds
    *threshold*.  Both noisy micro-segments and large false-detection jumps are
    skipped via min/max_seg guards.
    """
    for i in range(1, len(points) - 1):
        v1 = (points[i][0] - points[i-1][0], points[i][1] - points[i-1][1])
        v2 = (points[i+1][0] - points[i][0], points[i+1][1] - points[i][1])
        m1, m2 = math.hypot(*v1), math.hypot(*v2)
        # if m1 < min_seg or m2 < min_seg or m1 > max_seg or m2 > max_seg:
        #     continue
        angle = _angle_deg(v1, v2)
        print(f"Consecutive angle at {points[i]}: {angle:.2f}°")
        if angle >= threshold:
            print(f"Consecutive direction change at {points[i]}")
            return points[i]
    return None


def _anchored_change(
    points: list[tuple[int, int]],
    anchor: tuple[int, int],
    threshold: float = ANCHORED_ANGLE_THRESHOLD,
    min_seg: int = MIN_SEGMENT_PX,
    max_seg: int = MAX_SEGMENT_PX,
) -> tuple[int, int] | None:
    """Cumulative deviation from the initial post-pitch direction.

    Reference vector: anchor → points[0].
    Comparison vector: anchor → points[i].
    Returns the last stable point (points[i-1]) when drift exceeds *threshold*,
    so the caller gets the final "pre-deflection" position.
    """
    if len(points) < 2:
        return None
    v_ref = (points[0][0] - anchor[0], points[0][1] - anchor[1])
    print(f"Anchored reference vector between {anchor} and {points[0]}: {v_ref}")
    # if math.hypot(*v_ref) < min_seg:
    #     return None
    for i in range(1, len(points)):
        seg = math.hypot(points[i][0]-points[i-1][0], points[i][1]-points[i-1][1])
        # if seg > max_seg:          # skip jumps when building comparison vector
        #     continue
        v = (points[i][0] - anchor[0], points[i][1] - anchor[1])
        print(f"Anchored comparison vector at {points[i]}: {v}")
        # if math.hypot(*v) < min_seg:
        #     continue
        angle = _angle_deg(v_ref, v)
        print(f"Anchored angle at {points[i]}: {angle:.2f}°")
        if angle >= threshold:
            print(f"Anchored change — returning last stable point {points[i-1]}")
            return points[i-1]
    return None


def _pick_earliest(
    path: list[tuple[int, int]],
    *candidates: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Return the candidate appearing earliest in *path* (first contact wins).

    Candidates not found exactly in *path* are resolved by nearest-neighbour
    so the function never raises ValueError.
    """
    best_idx = len(path) - 1
    best_pt  = None
    for pt in candidates:
        if pt is None:
            continue
        try:
            idx = path.index(pt)
        except ValueError:
            idx = min(
                range(len(path)),
                key=lambda i, p=pt: math.hypot(path[i][0]-p[0], path[i][1]-p[1]),
            )
        if idx < best_idx:
            best_idx, best_pt = idx, pt
    return best_pt


# ── Public API ───────────────────────────────────────────────────────────────

def find_impact_point(
    ball_path_points:   list[tuple[int, int]],
    pitch_point:        tuple[int, int] | None,
    ball_in_bat_points: list[tuple[int, int]],
) -> tuple[int, int] | bool:
    """Locate the ball-bat impact point.

    Candidate priority (earliest in the ball path wins):
      1. First ball-inside-bat-bbox frame  — most direct evidence
      2. Y-axis turning point              — ball peak after pitch = bat contact
      3. Consecutive-vector angle change   — sharp deflection signal
      4. Anchored drift from post-pitch    — broader last-resort signal
      5. Last post-pitch point             — absolute fallback

    Full-toss (no pitch_point): candidates 1 & 3 only (no anchor / no Y-peak).
    """
    if not ball_path_points:
        return False

    bat_pt = ball_in_bat_points[-1] if ball_in_bat_points else None

    # ── Full-toss ─────────────────────────────────────────────────────────────
    if not pitch_point:
        dir_pt = _consecutive_change(ball_path_points)
        print(f"Full-toss — consecutive_change: {dir_pt}, bat: {bat_pt}")
        return _pick_earliest(ball_path_points, bat_pt, dir_pt) or False

    # ── Pitched delivery ──────────────────────────────────────────────────────
    try:
        pitch_idx = ball_path_points.index(pitch_point)
    except ValueError:
        pitch_idx = min(
            range(len(ball_path_points)),
            key=lambda i: math.hypot(
                ball_path_points[i][0] - pitch_point[0],
                ball_path_points[i][1] - pitch_point[1],
            ),
        )

    post_pitch = ball_path_points[pitch_idx + 1:]
    print(f"Post-pitch points ({len(post_pitch)}): {post_pitch}")

    if not post_pitch:
        return bat_pt or (ball_path_points[-1] if ball_path_points else False)

    # Bat points must come after the pitch to count
    post_pitch_set = set(post_pitch)
    post_bat_pt = next((p for p in ball_in_bat_points if p in post_pitch_set), None)

    turn_pt     = _turning_point(post_pitch)
    dir_pt      = _consecutive_change(post_pitch)
    anchored_pt = _anchored_change(post_pitch, pitch_point)

    print(f"Candidates bat: {post_bat_pt}, turn: {turn_pt}, "
          f"consecutive: {dir_pt}, anchored: {anchored_pt}")

    result = _pick_earliest(post_pitch, post_bat_pt, turn_pt, dir_pt, anchored_pt)
    return result or post_pitch[-1]

===
import math

# ── Thresholds ──────────────────────────────────────────────────────────────
ANGLE_CHANGE_THRESHOLD  = 25.0   # degrees — minimum deflection to count as a major angle change
MIN_GAP_BETWEEN_CHANGES = 40.0   # pixels  — prevents noise from creating a false 2nd change
PITCH_MATCH_TOLERANCE   = 40.0   # pixels  — max distance for pitch to "match" an angle-change point


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
                print(f"[full-toss guard] pitch {pitch_point} ≈ 2nd angle "
                      f"change {second_change} — treating as full-toss")
                return second_change

            # Standard bounce: 1st change ≈ pitch, 2nd change = impact
            return second_change

        if num_changes == 1:
            only_change = angle_changes[0]

            # If the single change IS the pitch, there's no bat impact
            if _distance(pitch_point, only_change) <= PITCH_MATCH_TOLERANCE:
                print(f"[pitch only] single angle change matches pitch — no bat impact")
                return bat_pt or False

            # Otherwise, the change is the bat impact
            return only_change

        # 0 angle changes — no bat contact detected
        return bat_pt or False

    # ── FULL-TOSS / no pitch ────────────────────────────────────────────
    if num_changes >= 1:
        return angle_changes[0]

    # Absolute fallback: use the last ball-inside-bat-bbox point
    return bat_pt or False

```

## Before vs After

| Before (196 lines, 4 sub-detectors) | After (128 lines, 1 unified algorithm) |
|---|---|
| `_turning_point` — Y-reversal | Removed (often confused pitch for impact) |
| `_consecutive_change` — angle > 30° | Replaced by [_find_angle_changes()](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py#28-59) with 25° threshold + 40px min gap |
| `_anchored_change` — cumulative drift > 45° | Removed (too broad, noisy) |
| `_pick_earliest` — earliest candidate wins | Replaced by case-based logic (pitch+changes → classify) |
| Segment filters commented out | No need — min-gap handles noise |

## Key Design Decisions

1. **Same public API preserved** — [find_impact_point(ball_path_points, pitch_point, ball_in_bat_points) → tuple | bool](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py#63-136)
2. **[pitch_point.py](file:///d:/CricVision_Scratch/ball_tracking/pitch_point.py) untouched** — orchestrator still calls it separately
3. **Bat-bbox kept as fallback** — used only when angle-change model produces no result
4. **Full-toss guard ported** — if pitch ≈ 2nd angle change (within 40px), it's a false pitch

## Verified

- ✅ Python syntax check passed
- ✅ Orchestrator's import (`from ball_tracking.impact_point import find_impact_point`) compatible
- ✅ [pitch_point.py](file:///d:/CricVision_Scratch/ball_tracking/pitch_point.py) not modified
