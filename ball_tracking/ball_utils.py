import math

# FOR STATIC BALLS

# Radius used both when building the static map (dedup) and when filtering
# live detections.  Matches the deployed pipeline.
STATIC_RADIUS = 60.0


def is_near_static(
    point: tuple[int, int],
    static_map: list[tuple[int, int]],
    threshold: float = STATIC_RADIUS,
) -> bool:
    px, py = point
    thresh_sq = threshold * threshold
    for sx, sy in static_map:
        if (px - sx) ** 2 + (py - sy) ** 2 < thresh_sq:
            return True
    return False


def build_static_ball_map(
    warmup_detections: list[list[tuple[int, int]]],
    cluster_radius: float = STATIC_RADIUS,
) -> list[tuple[int, int]]:
    """Collect every ball position seen during warmup as a static marker.

    Each detection is added to the map unless it falls within *cluster_radius*
    of an already-recorded entry (de-duplication).  No minimum-frame-count
    requirement — any ball visible during warmup is treated as stationary.
    """
    static_positions: list[tuple[int, int]] = []

    for frame_centers in warmup_detections:
        for cx, cy in frame_centers:
            if not is_near_static((cx, cy), static_positions, cluster_radius):
                static_positions.append((cx, cy))

    print(f"Identified {len(static_positions)} static ball positions: {static_positions}")
    return static_positions