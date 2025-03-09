import os
import json
import cv2
import re

# Define directories
data_dir = "data"
videos_dir = os.path.join(data_dir, "videos")
captions_dir = os.path.join(data_dir, "captions")

# Function to extract YouTube video ID from filename
def extract_video_id(filename):
    match = re.search(r"\[([A-Za-z0-9_-]+)\]", filename)  # Extracts text inside brackets [videoID]
    return match.group(1) if match else None

# Get first available video file
video_files = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")]
if not video_files:
    print("No video files found!")
    exit()

video_file = video_files[0]  # Grab the first video
video_path = os.path.join(videos_dir, video_file)

# Find corresponding caption file
video_id = extract_video_id(video_file)
if not video_id:
    print(f"Could not extract video ID from {video_file}")
    exit()

caption_file = next((f for f in os.listdir(captions_dir) if video_id in f and f.endswith(".json")), None)
if not caption_file:
    print(f"No matching caption file found for {video_file}")
    exit()

caption_path = os.path.join(captions_dir, caption_file)

# Load the caption JSON
with open(caption_path, "r", encoding="utf-8") as f:
    captions = json.load(f)

# Grab the first timestamp and its frames
if not captions:
    print("Caption file is empty!")
    exit()

first_entry = captions[0]
start_time = first_entry["start_time"]
end_time = first_entry["end_time"]
caption_text = first_entry["caption"]
frame_indices = first_entry.get("frames", [])

if not frame_indices:
    print("No frame indices found in the first caption entry!")
    exit()

print(f"\nSanity Check: {video_file}")
print(f"Caption: \"{caption_text}\"")
print(f"Timestamp: {start_time} → {end_time}")
print(f"Frame Indices: {frame_indices}")


cap = cv2.VideoCapture(video_path)

for frame_idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if ret:
        cv2.imshow(f"Frame {frame_idx}", frame)
        cv2.waitKey(500)  # Display each frame for 500ms
    else:
        print(f"❌ Failed to retrieve frame {frame_idx}")

cap.release()
cv2.destroyAllWindows()
