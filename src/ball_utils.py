import math

# FOR STATIC BALLS

def is_near_static(
    point: tuple[int, int],
    static_map: list[tuple[int, int]],
    threshold: float = 15.0,
) -> bool:
    px, py = point
    thresh_sq = threshold * threshold
    for sx, sy in static_map:
        if (px - sx) ** 2 + (py - sy) ** 2 < thresh_sq:
            return True
    return False


def build_static_ball_map(
    warmup_detections: list[list[tuple[int, int]]],
    min_frames: int = 2,
    cluster_radius: float = 15.0,
) -> list[tuple[int, int]]:
    
    all_ball_centers_flattened = [
        (frame_index, center_x, center_y)
        for frame_index, ball_centers_in_frame in enumerate(warmup_detections)
        for center_x, center_y in ball_centers_in_frame
    ]

    static_ball_positions: list[tuple[int, int]] = []
    
    is_ball_processed = [False] * len(all_ball_centers_flattened)

    for current_index, (frame_index, center_x, center_y) in enumerate(all_ball_centers_flattened):
        
        if is_ball_processed[current_index]:
            continue
        
        unique_frames_where_ball_appeared = {frame_index}
        
        for other_index, (other_frame_index, other_x, other_y) in enumerate(all_ball_centers_flattened):
            
            if other_index == current_index or is_ball_processed[other_index]:
                continue
            
            if math.hypot(center_x - other_x, center_y - other_y) < cluster_radius:
                unique_frames_where_ball_appeared.add(other_frame_index)
        
        if len(unique_frames_where_ball_appeared) >= min_frames:
            static_ball_positions.append((center_x, center_y))
            is_ball_processed[current_index] = True

    return static_ball_positions