import cv2
import numpy as np
import os
import math
# from storage_services import azure_storage
# from scripts.utils import add_watermark_img, save_video_ffmpeg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===== CONFIG =====
BALL_CONF_THRESHOLD = 0.70
INITIAL_CONF_THRESHOLD = 0.80
MAX_DISTANCE_THRESHOLD = 400
GRADIENT_ALPHA = 0.9
COLOR_START = (255, 255, 255)
COLOR_END = (255, 200, 0)

# Stationary ball detection config
STATIONARY_FRAMES = 5  # Number of initial frames to identify stationary balls
STATIONARY_DISTANCE_THRESHOLD = 15  # Max movement to consider a ball stationary
MOVING_BALL_MIN_DISTANCE = 20  # Minimum movement to identify as thrown ball


def get_center(x1, y1, x2, y2):
    """Get center point of bounding box."""
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def is_stationary_ball(center, stationary_balls, threshold=STATIONARY_DISTANCE_THRESHOLD):
    """Check if a detected ball is near any known stationary ball."""
    for stat_ball in stationary_balls:
        if euclidean_distance(center, stat_ball) < threshold:
            return True
    return False


def find_closest_to_last_position(detections, last_position, max_distance=MAX_DISTANCE_THRESHOLD, stationary_balls=None):
    """Find the detection closest to the last tracked position, excluding stationary balls."""
    if last_position is None or detections is None or len(detections) == 0:
        return None
    
    best_detection = None
    min_distance = float('inf')
    
    for detection in detections:
        x1, y1, x2, y2, conf, _ = detection
        center = get_center(float(x1), float(y1), float(x2), float(y2))
        
        # Skip if this is a stationary ball
        if stationary_balls and is_stationary_ball(center, stationary_balls):
            continue
        
        dist = euclidean_distance(center, last_position)
        if dist < min_distance and dist < max_distance:
            min_distance = dist
            best_detection = detection
    
    return best_detection


def process_video(input_video_path, output_video_path, obj_ball, obj_bat, max_distance_threshold=MAX_DISTANCE_THRESHOLD): 

    """ This function processes a cricket video to track the ball and detect impacts with the bat. 
    It filters out stationary balls and tracks only the thrown/moving ball.
    Returns a list of all tracked ball points for further analysis. """

    frame_count = 0
    
    # Storage for all ball trajectory points (frame_number, center_x, center_y, confidence)
    all_ball_points = []
    
    # Stationary ball tracking
    initial_ball_positions = []  # Store ball positions from first few frames
    stationary_balls = []  # Confirmed stationary ball positions
    
    # Moving ball tracking
    tracked_ball_position = None  # Last known position of the thrown ball
    thrown_ball_detected = False

    cap = cv2.VideoCapture(input_video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"Processing video: {total_frames} frames at {fps} fps")

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        frame_count += 1
        final_raw_frame = frame.copy()

        # --- BALL DETECTION ---
        ball_results = obj_ball.detect_ball(frame, confidence=BALL_CONF_THRESHOLD)
        ball_detections = ball_results[0].boxes.data if ball_results is not None and len(ball_results) > 0 and ball_results[0].boxes is not None else None

        current_detections = []
        if ball_detections is not None and len(ball_detections) > 0:
            for detection in ball_detections:
                x1, y1, x2, y2, conf, _ = detection
                center = get_center(float(x1), float(y1), float(x2), float(y2))
                current_detections.append({
                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                    'center': center,
                    'conf': float(conf),
                    'raw': detection
                })

        # --- PHASE 1: Identify stationary balls in first few frames ---
        if frame_count <= STATIONARY_FRAMES:
            for det in current_detections:
                initial_ball_positions.append({
                    'frame': frame_count,
                    'center': det['center'],
                    'conf': det['conf']
                })
            
            # After STATIONARY_FRAMES, identify which balls are stationary
            if frame_count == STATIONARY_FRAMES:
                # Group positions by proximity
                position_groups = []
                for pos in initial_ball_positions:
                    found_group = False
                    for group in position_groups:
                        if euclidean_distance(pos['center'], group['positions'][0]) < STATIONARY_DISTANCE_THRESHOLD:
                            group['positions'].append(pos['center'])
                            group['frames'].append(pos['frame'])
                            found_group = True
                            break
                    if not found_group:
                        position_groups.append({
                            'positions': [pos['center']],
                            'frames': [pos['frame']]
                        })
                
                # Balls appearing in multiple frames at same position are stationary
                for group in position_groups:
                    if len(group['frames']) >= STATIONARY_FRAMES - 1:
                        avg_x = sum(p[0] for p in group['positions']) / len(group['positions'])
                        avg_y = sum(p[1] for p in group['positions']) / len(group['positions'])
                        stationary_balls.append((int(avg_x), int(avg_y)))
                
                print(f"\n[INFO] Identified {len(stationary_balls)} stationary balls at positions: {stationary_balls}")

        # --- PHASE 2: Track the thrown ball (excluding stationary balls) ---
        thrown_ball_center = None
        thrown_ball_conf = None
        
        if frame_count > STATIONARY_FRAMES and current_detections:
            # Filter out stationary balls
            moving_detections = [
                det for det in current_detections 
                if not is_stationary_ball(det['center'], stationary_balls)
            ]
            
            if moving_detections:
                if tracked_ball_position is None:
                    # First moving ball detection - pick the one with highest confidence
                    best_det = max(moving_detections, key=lambda x: x['conf'])
                    if best_det['conf'] >= INITIAL_CONF_THRESHOLD:
                        thrown_ball_center = best_det['center']
                        thrown_ball_conf = best_det['conf']
                        tracked_ball_position = thrown_ball_center
                        thrown_ball_detected = True
                else:
                    # Find closest moving ball to last position
                    raw_detections = [det['raw'] for det in moving_detections]
                    if raw_detections:
                        closest = find_closest_to_last_position(
                            raw_detections, 
                            tracked_ball_position, 
                            max_distance_threshold,
                            stationary_balls
                        )
                        if closest is not None:
                            x1, y1, x2, y2, conf, _ = closest
                            thrown_ball_center = get_center(float(x1), float(y1), float(x2), float(y2))
                            thrown_ball_conf = float(conf)
                            tracked_ball_position = thrown_ball_center

        # --- STORE BALL POINT ---
        if thrown_ball_center is not None:
            all_ball_points.append({
                'frame': frame_count,
                'center': thrown_ball_center,
                'confidence': thrown_ball_conf
            })
            
            # --- DRAW DOT ON THROWN BALL ---
            cv2.circle(final_raw_frame, thrown_ball_center, 12, (0, 255, 0), -1)  # Green filled circle
            cv2.circle(final_raw_frame, thrown_ball_center, 14, (255, 255, 255), 2)  # White border
            
            # Optional: Draw trail of last few positions
            recent_points = all_ball_points[-15:]  # Last 15 points for trail
            for i, pt in enumerate(recent_points[:-1]):
                alpha = (i + 1) / len(recent_points)
                color = (
                    int(COLOR_START[0] * (1 - alpha) + COLOR_END[0] * alpha),
                    int(COLOR_START[1] * (1 - alpha) + COLOR_END[1] * alpha),
                    int(COLOR_START[2] * (1 - alpha) + COLOR_END[2] * alpha)
                )
                cv2.circle(final_raw_frame, pt['center'], int(5 + 3 * alpha), color, -1)

        # --- Mark stationary balls (optional - for debugging) ---
        for stat_ball in stationary_balls:
            cv2.circle(final_raw_frame, stat_ball, 8, (128, 128, 128), 1)  # Gray circle for stationary
            cv2.putText(final_raw_frame, "S", (stat_ball[0] - 5, stat_ball[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        # Write frame to output video
        out.write(final_raw_frame)
        
        print(f"\r[Frame {frame_count}/{total_frames}] Tracked ball: {thrown_ball_center}", end="")

    cap.release()
    out.release()
    
    print(f"\n\n[COMPLETE] Video saved to: {output_video_path}")
    print(f"[INFO] Total ball points tracked: {len(all_ball_points)}")
    
    # Return all ball points for further analysis
    return all_ball_points