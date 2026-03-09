def find_pitch_point(ball_path_points: list[tuple[int, int]]) -> tuple[int, int] | bool:
    """Heuristic to determine if a ball track likely represents a pitch point."""
    if len(ball_path_points) < 3:
        return False

    # Check for significant vertical drop followed by a change in direction
    for i in range(1, len(ball_path_points) - 1):
        prev_y = ball_path_points[i - 1][1]
        curr_y = ball_path_points[i][1]
        next_y = ball_path_points[i + 1][1]

        # Look for a significant drop (e.g., >20 pixels) followed by a change in direction
        if curr_y >= prev_y and next_y <= curr_y:
            return ball_path_points[i]

    return False