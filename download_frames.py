import os
import json
import cv2
import numpy as np
import re

# Define directories
data_dir = "data"
videos_dir = os.path.join(data_dir, "videos")
captions_dir = os.path.join(data_dir, "captions")

# Ensure directories exist
os.makedirs(videos_dir, exist_ok=True)
os.makedirs(captions_dir, exist_ok=True)

# Function to convert timestamp "hh:mm:ss,ms" to seconds
def timestamp_to_seconds(timestamp):
    h, m, s = timestamp.replace(',', '.').split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

# Function to get the video's frame rate (FPS)
def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

# Function to extract the YouTube video ID from filenames
def extract_video_id(filename):
    match = re.search(r"\[([A-Za-z0-9_-]+)\]", filename)  # Extracts text inside brackets [videoID]
    return match.group(1) if match else None

# Function to calculate frame indices for each caption
def extract_frame_indices(video_path, caption_data, num_frames=5):
    fps = get_video_fps(video_path)
    if fps is None:
        return caption_data  # Skip this video if FPS couldn't be determined

    updated_captions = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps  # Total duration in seconds

    for entry in caption_data:
        start_sec = timestamp_to_seconds(entry["start_time"])
        end_sec = timestamp_to_seconds(entry["end_time"])

        if end_sec > video_duration:
            end_sec = video_duration  # Prevent going beyond video length

        # Convert to frame indices
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        if end_frame > frame_count:
            end_frame = frame_count - 1  # Adjust if beyond video length

        # Ensure enough frames exist
        if end_frame - start_frame >= num_frames:
            frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int).tolist()
        else:
            frame_indices = list(range(start_frame, min(start_frame + num_frames, end_frame + 1)))

        entry["frames"] = frame_indices
        updated_captions.append(entry)

    cap.release()
    return updated_captions

# Process videos and captions by matching YouTube IDs
video_files = {extract_video_id(f): f for f in os.listdir(videos_dir) if extract_video_id(f) and f.endswith(".mp4")}
caption_files = {extract_video_id(f): f for f in os.listdir(captions_dir) if extract_video_id(f) and f.endswith(".json")}

for video_id, video_file in video_files.items():
    if video_id in caption_files:
        video_path = os.path.join(videos_dir, video_file)
        caption_file = caption_files[video_id]
        caption_path = os.path.join(captions_dir, caption_file)

        print(f"Processing: {video_file} -> {caption_file}")

        with open(caption_path, "r", encoding="utf-8") as f:
            captions = json.load(f)

        # Extract frame indices
        captions_with_frames = extract_frame_indices(video_path, captions)

        # Save updated captions
        with open(caption_path, "w", encoding="utf-8") as f:
            json.dump(captions_with_frames, f, indent=4, ensure_ascii=False)
    else:
        print(f"WARNING: No matching caption file found for video ID {video_id}")

print("Frame index extraction complete! JSON files now include frame indices.")
