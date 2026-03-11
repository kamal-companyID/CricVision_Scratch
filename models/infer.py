"""infer — YOLO model loading and inference wrapper."""

import numpy as np
import torch
from ultralytics import YOLO

from ball_tracking.logger import get_logger

logger = get_logger(__name__)


class Inference:
    """Thin wrapper around a YOLO model for ball / bat detection."""

    # ── Initialisation ───────────────────────────────────────────────────────

    def __init__(self) -> None:
        self.model: YOLO | None = None

    # ── Model Loading ─────────────────────────────────────────────────────────

    def load_model(self, model_path: str) -> None:
        """Load a YOLO model from *model_path* onto the best available device."""
        try:
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = YOLO(model_path).to(device)
            logger.info("Model loaded on %s: %s", device.upper(), model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model: {exc}") from exc

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, confidence: float = 0.5) -> list | None:
        """Run inference; return results only if exactly 1 detection with keypoints."""
        try:
            results: list = self.model(frame, conf=confidence, verbose=False, batch=16)
            detections = results[0]
            if detections.boxes is None or len(detections.boxes) != 1:
                return None
            if detections.keypoints is None or len(detections.keypoints) != 1:
                return None
            return results
        except Exception as exc:
            raise RuntimeError(f"Inference failed: {exc}") from exc

    def detect_ball(self, frame: np.ndarray, confidence: float = 0.5) -> list:
        """Run detection and return raw YOLO results."""
        return self.model(frame, conf=confidence, verbose=False)
