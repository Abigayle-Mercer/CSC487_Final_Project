import os
import subprocess
import shutil
import re
import json
import cv2


data_dir = "data"
captions_dir = os.path.join(data_dir, "captions")
videos_dir = os.path.join(data_dir, "videos")


os.makedirs(captions_dir, exist_ok=True)
os.makedirs(videos_dir, exist_ok=True)

# yt-dlp command
yt_dlp_command = [
    "yt-dlp",
    "--write-auto-sub",
    "--sub-lang", "en",
    "--convert-subs", "srt",
    "--skip-download",
    "https://www.youtube.com/watch?v=K-J0ticRXnM&list=PLw_rVnBBP57VHUSI1k2PgMQFFbDqxvSYZ&index=1"
]

# Run yt-dlp command with error handling
print("Downloading captions...")
result = subprocess.run(yt_dlp_command, capture_output=True, text=True)


# Move .srt files to captions directory
for file in os.listdir():
    if file.endswith(".srt"):
        shutil.move(file, os.path.join(captions_dir, file))

print("Download complete. Converting .srt files to .json...")

# Regular expression to match non-informative captions like [Music], [Applause], [Laughter]
non_informative_pattern = re.compile(r"\[.*?\]")

# Function to convert timestamp "hh:mm:ss,ms" to seconds
def timestamp_to_seconds(timestamp):
    h, m, s = timestamp.replace(',', '.').split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

# Function to determine frame rate from video
def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 30  # Default to 30 FPS if the video cannot be opened
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

# Function to parse an SRT file and filter captions with durations <= 5 frames
def parse_srt(file_path, fps):
    with open(file_path, "r", encoding="utf-8") as file:
        srt_data = file.read().split("\n\n")  # Split by double newlines (SRT entry separator)

    captions = []
    for entry in srt_data:
        lines = entry.split("\n")
        if len(lines) >= 3:
            start_time, end_time = lines[1].split(" --> ")  # Extract both timestamps
            caption = " ".join(lines[2:]).strip()

            # Remove non-informative tags like [Music]
            cleaned_caption = non_informative_pattern.sub("", caption).strip()

            # Convert timestamps to seconds
            start_sec = timestamp_to_seconds(start_time)
            end_sec = timestamp_to_seconds(end_time)
            duration_sec = end_sec - start_sec

            # Convert duration to frames
            duration_frames = duration_sec * fps

            # Ignore captions that are too short (≤ 5 frames)
            if cleaned_caption and duration_frames > 5:
                captions.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "caption": cleaned_caption
                })

    return captions

# Process all SRT files in the captions directory
for filename in os.listdir(captions_dir):
    if filename.endswith(".srt"):
        srt_path = os.path.join(captions_dir, filename)
        json_path = os.path.splitext(srt_path)[0] + ".json"

        # match the video filename to get FPS
        video_filename = filename.replace(".srt", ".mp4")
        video_path = os.path.join(videos_dir, video_filename)
        fps = get_video_fps(video_path)  # Get FPS from video

        # save JSON
        captions = parse_srt(srt_path, fps)
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(captions, json_file, indent=4, ensure_ascii=False)

        # Remove original SRT file
        os.remove(srt_path)

print("SRT files successfully converted to JSON and filtered.")
