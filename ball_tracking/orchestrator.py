import os
import tempfile
import time
import cv2
from ball_tracking.detections import detect_all, draw_detections
from ball_tracking.ball_utils import build_static_ball_map, is_near_static
from ball_tracking.pitch_point import find_pitch_point
from ball_tracking.impact_point import find_impact_point
from ball_tracking.ball_path import compute_full_path
from ball_tracking.drawing import draw_ball_path, draw_ball_path_animated
from ball_tracking.logger import get_logger
from ball_tracking.config import (
    WARMUP_FRAMES, STATIC_THRESHOLD, FREEZE_DURATION,
    INITIAL_CONF_THRESHOLD, INITIAL_CONF_FRAMES,
)

logger = get_logger(__name__)


def _find_impact_frame_idx(
    ball_path_points: list[tuple[int, int]],
    impact_point,
    frame_ball_map: dict[int, tuple[int, int]],
) -> int | None:
    """Return the frame index whose detected ball centre matches the impact point."""
    if not impact_point:
        return None
    for fidx, centre in frame_ball_map.items():
        if centre == impact_point:
            return fidx
    # Fallback: if impact_point came from ball_in_bat_points, it might be the
    # last entry — find closest frame.
    if frame_ball_map:
        return min(
            frame_ball_map.keys(),
            key=lambda f: (
                (frame_ball_map[f][0] - impact_point[0]) ** 2
                + (frame_ball_map[f][1] - impact_point[1]) ** 2
            ),
        )
    return None


def process_video(video_path: str, output_path: str | None = None, confidence: float = 0.5) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error("Error opening video file: %s", video_path)
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Pass 1: collect detections + write annotated frames to temp video ─────
    t0 = time.perf_counter()

    warmup_ball_centers: list[list[tuple[int, int]]] = []
    static_ball_map: list[tuple[int, int]] = []
    ball_path_points: list[tuple[int, int]] = []
    ball_in_bat_points: list[tuple[int, int]] = []
    frame_ball_map: dict[int, tuple[int, int]] = {}   # frame_idx → ball centre

    # Write annotated frames to a temporary video instead of storing in RAM
    temp_fd, temp_path = tempfile.mkstemp(suffix='.avi')
    os.close(temp_fd)
    temp_writer = cv2.VideoWriter(
        temp_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height),
    )

    frame_idx = 0
    tracked_frames = 0   # counts frames added to ball_path_points
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
                logger.info("Static ball map built with %d positions: %s", len(static_ball_map), static_ball_map)
        else:
            # Apply stricter confidence for the first INITIAL_CONF_FRAMES tracked frames
            min_conf = INITIAL_CONF_THRESHOLD if tracked_frames < INITIAL_CONF_FRAMES else confidence
            detections['ball'] = [
                d for d in detections['ball']
                if d['confidence'] >= min_conf
                and not is_near_static((d['cx'], d['cy']), static_ball_map, STATIC_THRESHOLD)
            ]
            if len(detections['ball']) > 1:
                detections['ball'] = [max(detections['ball'], key=lambda d: d['confidence'])]
            for d in detections['ball']:
                pt = (d['cx'], d['cy'])
                ball_path_points.append(pt)
                frame_ball_map[frame_idx] = pt
                tracked_frames += 1

            bats = detections['bat']
            if bats:
                bat = max(bats, key=lambda b: b['confidence'])
                for d in detections['ball']:
                    cx, cy = d['cx'], d['cy']
                    if bat['x1'] <= cx <= bat['x2'] and bat['y1'] <= cy <= bat['y2']:
                        ball_in_bat_points.append((cx, cy))

        temp_writer.write(draw_detections(frame, detections))
        frame_idx += 1

    cap.release()
    temp_writer.release()

    t1 = time.perf_counter()
    logger.info("Pass 1 (detection): %.2f s — %d frames", t1 - t0, frame_idx)

    # ── Compute key points & physics path ─────────────────────────────────────
    pitch_point  = find_pitch_point(ball_path_points) 
    impact_point = find_impact_point(ball_path_points, pitch_point, ball_in_bat_points) or (ball_in_bat_points[-1] if ball_in_bat_points else None) or ball_path_points[-1] if ball_path_points else None
    first_point  = ball_path_points[0] if ball_path_points else None

    logger.info("First point: %s, Pitch point: %s, Impact point: %s", first_point, pitch_point, impact_point)

    # # ── Fulltoss guard: impact must come AFTER pitch in the ball path ─────────
    if pitch_point and impact_point:
        try:
            pitch_idx_in_path  = ball_path_points.index(pitch_point)
        except ValueError:
            pitch_idx_in_path  = None
        try:
            impact_idx_in_path = ball_path_points.index(impact_point)
        except ValueError:
            impact_idx_in_path = None

        if pitch_idx_in_path is not None and impact_idx_in_path is not None:
            if impact_idx_in_path <= pitch_idx_in_path:
                # Impact before pitch → fulltoss, discard pitch
                logger.info("[fulltoss] Impact detected before pitch — discarding pitch point.")
                pitch_point = False

    if not pitch_point and impact_point:
        new_impact_point = find_impact_point(ball_path_points, pitch_point, ball_in_bat_points) or (ball_in_bat_points[-1] if ball_in_bat_points else None) or ball_path_points[-1] if ball_path_points else None
        logger.info("new impact point after fulltoss check: %s", new_impact_point)

    path_data = compute_full_path(first_point, pitch_point or None, impact_point or None)
    impact_fidx = _find_impact_frame_idx(ball_path_points, impact_point, frame_ball_map)

    t2 = time.perf_counter()
    logger.info("Computation: %.2f s", t2 - t1)

    # ── Pass 2: write output video with freeze-frame animation ────────────────
    writer = None
    if output_path:
        temp_cap = cv2.VideoCapture(temp_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if writer:
        freeze_frames = int(fps * FREEZE_DURATION)  # how many frames = 1 sec
        path_drawn = False                           # flipped once freeze done

        for idx in range(frame_idx):
            ret, aframe = temp_cap.read()
            if not ret:
                break

            # After the freeze is over, stamp the path onto every remaining frame
            if path_drawn:
                aframe = draw_ball_path(
                    aframe, path_data,
                    first_point,
                    pitch_point or None,
                    impact_point or None,
                )

            writer.write(aframe)

            # After writing the impact frame, insert the animated freeze
            if impact_fidx is not None and idx == impact_fidx:
                for f in range(freeze_frames):
                    progress = (f + 1) / freeze_frames        # 0 → 1
                    overlay = aframe.copy()
                    overlay = draw_ball_path_animated(
                        overlay, path_data, progress,
                        first_point,
                        pitch_point or None,
                        impact_point or None,
                    )
                    writer.write(overlay)
                path_drawn = True

        temp_cap.release()
        writer.release()

    # Clean up temporary file
    try:
        os.unlink(temp_path)
    except OSError:
        pass

    t3 = time.perf_counter()
    logger.info("Pass 2 (writing): %.2f s", t3 - t2)

    cv2.destroyAllWindows()

    logger.info("--- Results for %s ---", video_path)
    logger.info("total ball detections: %d", len(ball_path_points))
    logger.debug("valid ball detections (after filtering static): %s", ball_path_points)
    logger.debug("ball points inside bat bbox: %s", ball_in_bat_points)
    logger.info("First point:  %s", first_point)
    logger.info("Pitch point:  %s", pitch_point)
    logger.info("Impact point: %s", impact_point)
    logger.info("Done — processed %d frames in %.2f s total.", frame_idx, t3 - t0)
    logger.debug("Static ball map: %s", static_ball_map)