import os
import json
from datetime import datetime
# from ball_track_deviation import process_video
from my_code import process_video
from models import obj_ball, obj_bat

input_root = r"D:\CricVision_manual_code_new\inputs"
# input_root = r"c:\Users\Admin\Downloads\user_videos\n"
# input_root = r"c:\Users\Admin\Downloads\user_videos\jussie_graham_clips\clips"
# input_root = r"c:\Users\Admin\Downloads\user_videos\user2"
# input_root = r"d:\ball-tracking\clips\vadodara\test"
# input_root = r"d:\ball-tracking\clips\Udaipur\multiple"
# input_root = r"d:\ball-tracking\clips\clips\bad impact\temp"
# input_root = r"d:\ball-tracking\clips\clips\good_pitch_impact\temp\temp\angle"
# input_root = r"c:\Users\Admin\Downloads\testing_8jan"
# input_root = r"d:\ball-tracking\clips\full_Pitch"
# input_root = r"d:\ball-tracking\clips\test_ratio"
video_files = []
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(('.mp4', '.mov', '.avi')):
            full_path = os.path.join(root, file)
            video_files.append(full_path)

for input_path in video_files:
    print(f"\nProcessing: {input_path}")
    
    # Get relative path from root (to preserve hierarchy)
    relative_path = os.path.relpath(input_path, input_root)
    input_name, ext = os.path.splitext(os.path.basename(input_path))
    today_str = datetime.now().strftime("%Y-%m-%d")
    # output_root = os.path.join("output_ball_track_only", today_str)
    # output_root = os.path.join("output_multiple", today_str)    
    output_root = os.path.join("outputs", today_str)    
    
    # Construct the full output path (preserve subfolder structure)
    output_subfolder = os.path.dirname(relative_path)
    output_dir = os.path.join(output_root, output_subfolder)
    os.makedirs(output_dir, exist_ok=True)
    
    # Final output video path
    output_video_path = os.path.join(output_dir, f"{input_name}.mp4")
    # process_video(input_path, output_video_path)
    ball_points = process_video(input_path, output_video_path, obj_ball, obj_bat)
    
    # Save ball points to JSON for further analysis
    ball_points_path = os.path.join(output_dir, f"{input_name}_ball_points.json")
    with open(ball_points_path, 'w') as f:
        json.dump(ball_points, f, indent=2)
    print(f"[INFO] Ball trajectory points saved to: {ball_points_path}")