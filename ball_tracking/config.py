"""
config.py — Central configuration for all CricVision constants.

Adjust any value here and it will take effect across the entire pipeline.
"""

import os

# ═════════════════════════════════════════════════════════════════════════════
#  INPUT / OUTPUT
# ═════════════════════════════════════════════════════════════════════════════
INPUT_ROOT = r"D:\CricVision_Scratch\inputs\New folder"  # Change this to your input folder

# ═════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═════════════════════════════════════════════════════════════════════════════
ROOT_LOGGER_NAME = "CricVision"
LOG_DIR          = "logs"
LOG_FILE         = os.path.join(LOG_DIR, "cricvision.log")
LOG_MAX_BYTES    = 5 * 1024 * 1024   # 5 MB per log file
LOG_BACKUP_COUNT = 5
LOG_FILE_FMT     = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
LOG_CONSOLE_FMT  = "[%(asctime)s] [%(levelname)-8s] %(message)s"
LOG_DATE_FMT     = "%Y-%m-%d %H:%M:%S"

# ═════════════════════════════════════════════════════════════════════════════
#  DETECTION TOGGLES
# ═════════════════════════════════════════════════════════════════════════════
DETECT_BALL  = True
DETECT_BAT   = True
DETECT_STUMP = False

# ═════════════════════════════════════════════════════════════════════════════
#  DETECTION BOX COLOURS (BGR)
# ═════════════════════════════════════════════════════════════════════════════
DETECTION_COLORS = {
    'ball':  (0, 255, 0),
    'bat':   (255, 128, 0),
    'stump': (0, 0, 255),
}

# ═════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR / PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
WARMUP_FRAMES          = 2
STATIC_THRESHOLD       = 60.0    # must match STATIC_RADIUS
FREEZE_DURATION        = 1.0     # seconds to freeze after impact frame
INITIAL_CONF_THRESHOLD = 0.80    # min confidence for first N tracked frames
INITIAL_CONF_FRAMES    = 3       # how many frames the stricter threshold applies

# ═════════════════════════════════════════════════════════════════════════════
#  STATIC BALL FILTERING (ball_utils)
# ═════════════════════════════════════════════════════════════════════════════
STATIC_RADIUS = 60.0

# ═════════════════════════════════════════════════════════════════════════════
#  IMPACT POINT / ANGLE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
ANGLE_CHANGE_THRESHOLD  = 25.0   # degrees — minimum deflection for major angle change
MIN_GAP_BETWEEN_CHANGES = 40.0   # pixels  — prevents noise from creating false 2nd change
PITCH_MATCH_TOLERANCE   = 40.0   # pixels  — max distance for pitch to "match" angle-change point

# ═════════════════════════════════════════════════════════════════════════════
#  BALL PATH / PHYSICS ARC
# ═════════════════════════════════════════════════════════════════════════════
DELIVERY_SAG_FACTOR = 0.10       # free-fall: arc sags downward
BOUNCE_SAG_FACTOR   = 0.12       # bounce: arc bulges upward (negated internally)
DEFAULT_N_POINTS    = 60         # sampled curve points per arc

# ═════════════════════════════════════════════════════════════════════════════
#  DRAWING / VISUAL OVERLAY
# ═════════════════════════════════════════════════════════════════════════════
DRAW_REF_SIZE      = 1080
DRAW_SCALE_MIN     = 0.6
DRAW_SCALE_MAX     = 1.6

GRAD_START         = (255, 255, 255)     # gradient start colour (BGR)
GRAD_END           = (255, 200, 0)       # gradient end colour (BGR)

PITCH_POINT_COLOR  = (255, 0, 0)         # blue in BGR
IMPACT_POINT_COLOR = (0, 0, 255)         # red in BGR

PATH_ALPHA         = 0.8                 # transparency for path overlay
