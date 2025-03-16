#imports
import os
import json
import cv2
import re
import torch
import clip
from PIL import Image
import torchvision.transforms as transforms
import datetime
import torch.nn.functional as F
import random

import tqdm


#consts
VIDS = [
    "data/videos/Cal Poly Survivor： S3 E1： Wit, Grit & Charm [dfYMJhtFvuU].mp4",
    "data/videos/Cal Poly Survivor： S3 E2： Not The Gentle Giant You Think He Is! [JtwZPIwDt8g].mp4",
    "data/videos/Cal Poly Survivor： S3 E3： My Mother Would Be Proud Of Me [AIUiNM3UYWE].mp4",
    "data/videos/Cal Poly Survivor： S3 E4： Fake Vibes [mh17q1buqJA].mp4",
    "data/videos/Cal Poly Survivor： S3 E5： David V Goliath [2vfBoG5hI1g].mp4",
    "data/videos/Cal Poly Survivor： S3 E6： I Haven't Been Invested [5J4eGzhAHSM].mp4",
    "data/videos/Cal Poly Survivor： S3 E7： Somebody Drop Their Buff？？ [1smUsfqs54s].mp4",
    "data/videos/Cal Poly Survivor： S3 E8： Like a Mob Boss [jJePD7jcNBQ].mp4",
    "data/videos/Cal Poly Survivor： S3 E9： Eating Challenge [odyJOeKpZjs].mp4",
    "data/videos/Cal Poly Survivor： S3 E10： Loved Ones [Kggc-m8ntVQ].mp4",
    "data/videos/Cal Poly Survivor： S3 E11： I'm Keeping It [bCQcuWrIm-c].mp4",
    "data/videos/Cal Poly Survivor： S3 E12： Super Idol! [K-J0ticRXnM].mp4",
]

CAPS = [
    "data/captions/Cal Poly Survivor： S3 E1： Wit, Grit & Charm [dfYMJhtFvuU].en.json",
    "data/captions/Cal Poly Survivor： S3 E2： Not The Gentle Giant You Think He Is! [JtwZPIwDt8g].en.json",
    "data/captions/Cal Poly Survivor： S3 E3： My Mother Would Be Proud Of Me [AIUiNM3UYWE].en.json",
    "data/captions/Cal Poly Survivor： S3 E4： Fake Vibes [mh17q1buqJA].en.json",
    "data/captions/Cal Poly Survivor： S3 E5： David V Goliath [2vfBoG5hI1g].en.json",
    "data/captions/Cal Poly Survivor： S3 E6： I Haven't Been Invested [5J4eGzhAHSM].en.json",
    "data/captions/Cal Poly Survivor： S3 E7： Somebody Drop Their Buff？？ [1smUsfqs54s].en.json",
    "data/captions/Cal Poly Survivor： S3 E8： Like a Mob Boss [jJePD7jcNBQ].en.json",
    "data/captions/Cal Poly Survivor： S3 E9： Eating Challenge [odyJOeKpZjs].en.json",
    "data/captions/Cal Poly Survivor： S3 E10： Loved Ones [Kggc-m8ntVQ].en.json",
    "data/captions/Cal Poly Survivor： S3 E11： I'm Keeping It [bCQcuWrIm-c].en.json",
    "data/captions/Cal Poly Survivor： S3 E12： Super Idol! [K-J0ticRXnM].en.json",
]

#helper functions:
def time_to_seconds(time_str):
    """Convert timestamp from 'HH:MM:SS,ms' to total seconds."""
    time_obj = datetime.datetime.strptime(time_str, "%H:%M:%S,%f")
    return (time_obj.hour * 3600) + (time_obj.minute * 60) + time_obj.second + (time_obj.microsecond / 1_000_000)

def find_caption_at_time(json_file, query_time):
    """Finds the caption at a specific query time."""
    
    # Load JSON data
    with open(json_file, "r", encoding="utf-8") as f:
        captions = json.load(f)

    # Convert query time to seconds
    query_seconds = time_to_seconds(query_time)

    # Search for the matching caption
    for entry in captions:
        start_seconds = time_to_seconds(entry["start_time"])
        end_seconds = time_to_seconds(entry["end_time"])
        
        if start_seconds <= query_seconds <= end_seconds:
            return entry  # Return full caption entry

    return None  # No matching caption found

def clip_dot(image_features, text_features, temperature=0.07):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logits_per_image = image_features @ text_features.T
    #logits_per_text = text_features @ image_features.T

    logits_per_image /= temperature
    #logits_per_text /= temperature

    return F.softmax(logits_per_image)

def compare_captions(timestamp, cap_list, cap_file, vid_file, include_real=True, show_frame=True):

    entry = find_caption_at_time(
        cap_file, timestamp
    )

    frame_indices = entry["frames"]
    caption_text = entry["caption"]

    # Lazy-load one random frame
    cap = cv2.VideoCapture(vid_file)
    frame_idx = random.choice(frame_indices)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        frame_tensor = torch.zeros((3, 224, 224), dtype=torch.float)
    else:
        cv2.imshow("Image", frame)
        cv2.waitKey()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frame_tensor = preprocess(pil_image).to(device)
    
    frame_tensor = frame_tensor.unsqueeze(0)

    if include_real:
        texts = [caption_text, *cap_list]
    else:
        texts = cap_list
    text_tokens = clip.tokenize(texts, truncate=True).to(device)

    return frame_tensor, text_tokens, texts

def use_model(ep_id, timestamp, caption_options, include_real=True):
    frame, text_tok, texts = compare_captions(
        timestamp,
        caption_options,
        CAPS[ep_id],
        VIDS[ep_id],
        include_real=include_real
    )

    image_features = model.encode_image(frame)
    text_features = model.encode_text(text_tok)

    for t, closeness in zip(texts, clip_dot(image_features, text_features).tolist()[0]):
        print(f"P(text): {round(closeness,3)} | '{t}'")

#load the model architecture
device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#load model from checkpoint
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

checkpoint_path = os.path.join(save_dir, f"clip_overnight.pt")
checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device=device)
model.eval()

if __name__ == "__main__":
    use_model(
        0, "00:19:05,000",
        ["I like Rachel", "I do not like Rachel",
         "I like Abby", "I do not like Abby",
         "I like Scott", "I do not like Scott",
         "I like Nick", "I do not like Nick"],
         include_real=False
    )