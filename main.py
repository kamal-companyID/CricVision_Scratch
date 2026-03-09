import os
from datetime import datetime
from src.orchestrator import process_video

# Set input folder path here
input_root = r"inputs"  # Change this to your input folder

# Create output folder
today_str = datetime.now().strftime("%Y-%m-%d")
output_root = os.path.join("outputs", today_str)
os.makedirs(output_root, exist_ok=True)

# Find all video files
video_files = []
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(('.mp4', '.mov', '.avi')):
            full_path = os.path.join(root, file)
            video_files.append(full_path)

if not video_files:
    print(f"No video files found in {input_root}")
else:
    for input_path in video_files:
        print(f"\nProcessing: {input_path}")
        
        # Get relative path from root (to preserve hierarchy)
        relative_path = os.path.relpath(input_path, input_root)
        input_name, ext = os.path.splitext(os.path.basename(input_path))
        
        # Construct the full output path (preserve subfolder structure)
        output_subfolder = os.path.dirname(relative_path)
        output_dir = os.path.join(output_root, output_subfolder)
        os.makedirs(output_dir, exist_ok=True)
        
        # Final output video path
        output_video_path = os.path.join(output_dir, f"{input_name}.mp4")
        
        ball_path_points, ball_in_bat_points = process_video(input_path, output_video_path)
        print(f"Output saved to: {output_video_path}")