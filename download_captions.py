import os
import subprocess
import shutil
import re
import json

# Define the directory structure
data_dir = "data"
captions_dir = os.path.join(data_dir, "captions")
frames_dir = os.path.join(data_dir, "frames")

# Create directories if they don't exist
os.makedirs(captions_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

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

# Check for errors but continue execution
if result.returncode != 0:
    print(f"Warning: yt-dlp encountered an issue but will continue.\nError output:\n{result.stderr}")

# Move .srt files to captions directory
for file in os.listdir():
    if file.endswith(".srt"):
        shutil.move(file, os.path.join(captions_dir, file))

print("Download complete. Converting .srt files to .json...")

# Regular expression to match non-informative captions like [Music], [Applause], [Laughter]
non_informative_pattern = re.compile(r"\[.*?\]")

# Function to parse an SRT file
def parse_srt(file_path):
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

            # Ignore empty captions after cleaning
            if cleaned_caption:
                captions.append({"start_time": start_time, "end_time": end_time, "caption": cleaned_caption})

    return captions

# Process all SRT files in the captions directory
for filename in os.listdir(captions_dir):
    if filename.endswith(".srt"):
        srt_path = os.path.join(captions_dir, filename)
        json_path = os.path.splitext(srt_path)[0] + ".json"

        # Convert and save JSON
        captions = parse_srt(srt_path)
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(captions, json_file, indent=4, ensure_ascii=False)

        # Remove original SRT file
        os.remove(srt_path)

print("SRT files successfully converted to JSON and removed.")
