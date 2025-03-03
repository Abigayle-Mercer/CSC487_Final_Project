import torch
import matplotlib.pyplot as plt

import torch

tensor_file = "data/tensors/Cal Poly Survivor： S3 E8： Like a Mob Boss [jJePD7jcNBQ].pt"

try:
    # Explicitly disable "weights only" mode
    data = torch.load(tensor_file, map_location="cpu", weights_only=False)

    print("✅ Successfully loaded the .pt file!")
    print(f"Keys: {data.keys()}")
    print(f"Frames shape: {data['frames'].shape}")  # Should be (N, 3, 224, 224)
    print(f"Timestamps count: {len(data['timestamps'])}")  # Should match caption count

except Exception as e:
    print(f"❌ Failed to load .pt file: {e}")


frames = data["frames"]  # (total_frames, 3, 224, 224)
timestamps = data["timestamps"]  # List of frame index lists

# Select a caption index
caption_idx = 0
frame_indices = timestamps[caption_idx]
print(f"📌 Caption {caption_idx} uses frame indices: {frame_indices}")

# Extract the frames
frames_for_caption = frames[frame_indices]  # Shape: (5, 3, 224, 224)

# Convert to NumPy and fix dtype
frames_for_caption = frames_for_caption.permute(0, 2, 3, 1).cpu().numpy()  # (5, 224, 224, 3)
frames_for_caption = frames_for_caption.astype("float32")  # ✅ Convert to float32

# Display the frames
fig, axes = plt.subplots(1, len(frames_for_caption), figsize=(15, 5))
for i, img in enumerate(frames_for_caption):
    axes[i].imshow(img * 0.5 + 0.5)  # Undo normalization for display
    axes[i].axis("off")
plt.show()
