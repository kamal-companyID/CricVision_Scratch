import math

ANGLE_THRESHOLD = 30  # degrees — minimum direction change to count as an impact


def _direction_change(
    points: list[tuple[int, int]],
    pitch_point: tuple[int, int] | None = None,
    threshold: float = ANGLE_THRESHOLD,
) -> tuple[int, int] | None:
    """Return the first point in *points* where the trajectory changes
    direction by at least *threshold* degrees, or None."""
    if len(points) < 3:
        return None
    for i in range(1, len(points) - 1):
        p0, p1, p2 = points[i - 1], points[i], points[i + 1]
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        mag1, mag2 = math.hypot(*v1), math.hypot(*v2)
        if mag1 == 0 or mag2 == 0:
            continue
        cos_a = max(-1.0, min(1.0, (v1[0]*v2[0] + v1[1]*v2[1]) / (mag1 * mag2)))
        if math.degrees(math.acos(cos_a)) >= threshold:
            print(f"Direction change at (fulltoss) : {p1}") if pitch_point is None else print(f"Direction change at: {p1}")
            return p1
    return None


def _pick_latest(
    path: list[tuple[int, int]],
    a: tuple[int, int] | None,
    b: tuple[int, int] | None,
) -> tuple[int, int] | bool:
    """Return whichever of *a* / *b* appears later in *path*.
    If only one is valid, return that.  If neither, return False."""
    idx_a = path.index(a) if a is not None else -1
    idx_b = path.index(b) if b is not None else -1
    if idx_a == -1 and idx_b == -1:
        return False
    if idx_a == -1:
        return b
    if idx_b == -1:
        return a
    return a if idx_a >= idx_b else b


def find_impact_point(ball_path_points: list[tuple[int, int]], pitch_point: tuple[int, int], ball_in_bat_points: list[tuple[int, int]]) -> tuple[int, int] | bool:
    if not ball_path_points:
        return False

    if not pitch_point:
        direction_change_pt = _direction_change(ball_path_points)
        bat_pt = ball_in_bat_points[-1] if ball_in_bat_points else None

        return _pick_latest(direction_change_pt, bat_pt)

    try:
        pitch_index = ball_path_points.index(pitch_point)
    except ValueError:
        pitch_index = min(
            range(len(ball_path_points)),
            key=lambda i: math.hypot(
                ball_path_points[i][0] - pitch_point[0],
                ball_path_points[i][1] - pitch_point[1],
            ),
        )

    post_pitch = ball_path_points[pitch_index + 1 :]

    return _direction_change(post_pitch, pitch_point) or (ball_in_bat_points[-1] if ball_in_bat_points else False)
    # # Need at least 3 points to measure a direction change (p0→p1 vs p1→p2)
    # if len(post_pitch) >= 3:
    #     for i in range(1, len(post_pitch) - 1):
    #         p0, p1, p2 = post_pitch[i - 1], post_pitch[i], post_pitch[i + 1]

    #         v1 = (p1[0] - p0[0], p1[1] - p0[1])
    #         v2 = (p2[0] - p1[0], p2[1] - p1[1])

    #         mag1 = math.hypot(*v1)
    #         mag2 = math.hypot(*v2)

    #         if mag1 == 0 or mag2 == 0:
    #             continue

    #         # Angle between the two consecutive direction vectors
    #         cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
    #         cos_angle = max(-1.0, min(1.0, cos_angle))  # numerical clamp
    #         angle_deg = math.degrees(math.acos(cos_angle))

    #         if angle_deg >= ANGLE_THRESHOLD:
    #             print("Direction change at: ", post_pitch[i])
    #             return post_pitch[i]

    # # Fallback: if no direction change detected, use first ball-in-bat point
    # if ball_in_bat_points:
    #     return ball_in_bat_points[-1]

    # return False
