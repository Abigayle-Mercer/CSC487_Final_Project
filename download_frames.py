import os
import torch
import cv2
import numpy as np
import json
from torchvision import transforms
from glob import glob
import re
from PIL import Image

# Paths
VIDEO_DIR = "data/videos/"
CAPTION_DIR = "data/captions/"
OUTPUT_DIR = "data/frames/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# CLIP Preprocessing
clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
])

def time_to_frames(time_str, fps=30):
    """Converts timestamp (HH:MM:SS,ms) to frame index."""
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", time_str)
    if not match:
        raise ValueError(f"Invalid time format: {time_str}")

    hours, minutes, seconds, milliseconds = map(int, match.groups())
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return int(total_seconds * fps)

def extract_frames(video_path, timestamps, num_frames=5):
    """Extracts frames from video at given timestamps, ensuring at least 5 frames per caption."""
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    frame_metadata = []

    for start_time, end_time in timestamps:
        start_idx = time_to_frames(start_time)
        end_idx = time_to_frames(end_time)

        # Skip if fewer than 5 unique frames can be extracted
        if (end_idx - start_idx) < num_frames:
            print(f"⏩ Skipping caption ({start_time} - {end_time}): Not enough frames ({end_idx - start_idx} available).")
            continue

        frame_idxs = np.linspace(start_idx, end_idx, num_frames, dtype=int)
        frame_idxs = sorted(set(frame_idxs))  # Ensure unique frames

        frames_for_caption = []
        print(f"⏳ Extracting {len(frame_idxs)} frames for caption: {start_time} - {end_time}")
        
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                tensor = clip_transform(frame)
                frames_for_caption.append(tensor)
            else:
                print(f"❌ Failed to extract frame {idx} from {video_path}")

        # Only save if exactly 5 frames were extracted
        if len(frames_for_caption) == num_frames:
            all_frames.extend(frames_for_caption)
            frame_metadata.append(frame_idxs)
            print(f"   ✅ Stored {len(frames_for_caption)} frames for caption: {start_time} - {end_time}")

    cap.release()
    
    if all_frames:
        print(f"✅ Total frames extracted from {video_path}: {len(all_frames)}")
        return torch.stack(all_frames), frame_metadata
    else:
        print(f"❌ No valid captions found for {video_path}")
        return None, None


def process_video(video_path):
    """Processes a video, extracts frames, and stores tensors with metadata efficiently."""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    caption_file = os.path.join(CAPTION_DIR, f"{video_id}.en.json")

    with open(caption_file, "r") as f:
        captions = json.load(f)

    timestamps = [(c["start_time"], c["end_time"]) for c in captions]

    # Extract frames
    frame_tensors, frame_metadata = extract_frames(video_path, timestamps)

    if frame_tensors is not None:
        # Convert to float16 (saves space)
        frame_tensors = frame_tensors.half()

        save_path = os.path.join(OUTPUT_DIR, f"{video_id}.pt")
        
        # Prevent corruption by forcing legacy PyTorch format
        torch.save(
            {"frames": frame_tensors, "timestamps": frame_metadata},
            save_path,
            _use_new_zipfile_serialization=False  # ✅ Fixes corruption issue
        )

        print(f"✅ Successfully saved: {save_path} with {frame_tensors.shape[0]} frames")
    else: 
        print("❌ No valid frames extracted, skipping save.")

# Process all videos
for video_file in glob(os.path.join(VIDEO_DIR, "*.mkv")):
    print(f"📂 Processing: {video_file}")
    process_video(video_file)


