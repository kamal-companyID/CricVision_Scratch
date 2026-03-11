import os
from datetime import datetime
from ball_tracking.logger import get_logger
from ball_tracking.orchestrator import process_video

logger = get_logger(__name__)

# Set input folder path here
input_root = r"D:\CricVision_Scratch\inputs\New folder"  # Change this to your input folder

# Create output folder
today_str = datetime.now().strftime("%Y-%m-%d")
output_root = os.path.join("outputs", today_str)
os.makedirs(output_root, exist_ok=True)

# ── Startup banner ───────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("CricVision Pipeline — session started")
logger.info("  Timestamp  : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
logger.info("  Input root : %s", input_root)
logger.info("  Output root: %s", output_root)
logger.info("=" * 60)

# Find all video files
video_files = []
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(('.mp4', '.mov', '.avi')):
            full_path = os.path.join(root, file)
            video_files.append(full_path)

if not video_files:
    logger.warning("No video files found in %s", input_root)
else:
    logger.info("Found %d video(s) to process", len(video_files))
    for input_path in video_files:
        logger.info("Processing: %s", input_path)

        # Get relative path from root (to preserve hierarchy)
        relative_path = os.path.relpath(input_path, input_root)
        input_name, ext = os.path.splitext(os.path.basename(input_path))

        # Construct the full output path (preserve subfolder structure)
        output_subfolder = os.path.dirname(relative_path)
        output_dir = os.path.join(output_root, output_subfolder)
        os.makedirs(output_dir, exist_ok=True)

        # Final output video path
        output_video_path = os.path.join(output_dir, f"{input_name}.mp4")

        process_video(input_path, output_video_path)
        logger.info("Output saved to: %s", output_video_path)

logger.info("=" * 60)
logger.info("CricVision Pipeline — session finished")
logger.info("=" * 60)