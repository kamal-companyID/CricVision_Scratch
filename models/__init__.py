"""models — pre-loaded YOLO model instances for ball and bat detection."""

from models.infer import Inference

# ── Model Instances ─────────────────────────────────────────────────────────

obj_ball: Inference = Inference()
obj_bat: Inference = Inference()
obj_stump: Inference = Inference()

obj_ball.load_model("models/weights/ball_detection.pt")
obj_bat.load_model("models/weights/bat_detection.pt")
obj_stump.load_model("models/weights/stump_detection.pt")
