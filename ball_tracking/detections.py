import cv2
from models import obj_ball, obj_bat, obj_stump

DETECT_BALL = True
DETECT_BAT = True
DETECT_STUMP = False

_COLORS = {
    'ball':  (0, 255, 0),
    'bat':   (255, 128, 0),
    'stump': (0, 0, 255),
}


def _run(inference_obj, frame, confidence: float) -> list[dict]:
    results = inference_obj.model(frame, conf=confidence, verbose=False)
    detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cx': (x1 + x2) // 2, 'cy': (y1 + y2) // 2,
                'confidence': conf,
            })
    return detections


def detect_ball(frame, confidence: float = 0.65) -> list[dict]:
    if not DETECT_BALL:
        return []
    return _run(obj_ball, frame, confidence)


def detect_bat(frame, confidence: float = 0.5) -> list[dict]:
    if not DETECT_BAT:
        return []
    return _run(obj_bat, frame, confidence)


def detect_stump(frame, confidence: float = 0.5) -> list[dict]:
    if not DETECT_STUMP:
        return []
    return _run(obj_stump, frame, confidence)


def detect_all(frame, confidence: float = 0.65) -> dict[str, list[dict]]:
    return {
        'ball':  detect_ball(frame, confidence),
        'bat':   detect_bat(frame, confidence),
        'stump': detect_stump(frame, confidence),
    }


def draw_detections(frame, detections: dict[str, list[dict]]):
    annotated = frame.copy()
    for label, dets in detections.items():
        color = _COLORS.get(label, (255, 255, 255))
        for det in dets:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            conf = det['confidence']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated, text,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 2, cv2.LINE_AA,
            )
    return annotated
