import cv2
from ball_tracking.detections import detect_all, draw_detections
from ball_tracking.ball_utils import build_static_ball_map, is_near_static
from ball_tracking.pitch_point import find_pitch_point
from ball_tracking.impact_point import find_impact_point

WARMUP_FRAMES = 15
STATIC_THRESHOLD = 15.0


def process_video(video_path: str, output_path: str | None = None, confidence: float = 0.5) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    warmup_ball_centers: list[list[tuple[int, int]]] = []
    static_ball_map: list[tuple[int, int]] = []
    ball_path_points: list[tuple[int, int]] = []
    ball_in_bat_points: list[tuple[int, int]] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_all(frame, confidence)

        if frame_idx < WARMUP_FRAMES:
            centers = [(d['cx'], d['cy']) for d in detections['ball']]
            warmup_ball_centers.append(centers)
            if frame_idx == WARMUP_FRAMES - 1:
                static_ball_map = build_static_ball_map(warmup_ball_centers)
        else:
            detections['ball'] = [
                d for d in detections['ball']
                if not is_near_static((d['cx'], d['cy']), static_ball_map, STATIC_THRESHOLD)
            ]
            for d in detections['ball']:
                ball_path_points.append((d['cx'], d['cy']))

            bats = detections['bat']
            if bats:
                bat = max(bats, key=lambda b: b['confidence'])
                for d in detections['ball']:
                    cx, cy = d['cx'], d['cy']
                    if bat['x1'] <= cx <= bat['x2'] and bat['y1'] <= cy <= bat['y2']:
                        ball_in_bat_points.append((cx, cy))

        annotated_frame = draw_detections(frame, detections)
        
        if writer:
            writer.write(annotated_frame)

        frame_idx += 1
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    pitch_point = find_pitch_point(ball_path_points)
    impact_point = find_impact_point(ball_path_points, pitch_point, ball_in_bat_points)

    print(f"Impact point: {impact_point}")
    print(f"valid ball detections (after filtering static): {ball_path_points}")
    print(f"ball points inside bat bbox: {ball_in_bat_points}")
    print(f"Pitch point: {pitch_point}")
    print(f"Done — processed {frame_idx} frames.")
    return ball_path_points, ball_in_bat_points