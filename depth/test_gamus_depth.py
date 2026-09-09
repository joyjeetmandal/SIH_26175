import sys
import numpy as np
import torch

# Add project root to Python path
sys.path.append(".")

# Add Depth Anything V2 to Python path
sys.path.append("models/Depth-Anything-V2")

from depth_anything_v2.dpt import DepthAnythingV2
from datasets.gamus import GAMUSDataset


# -----------------------------
# 1. Load dataset
# -----------------------------

dataset = GAMUSDataset("data/Gamus")

sample = dataset[0]

rgb = sample["rgb"]

print("Original RGB:")
print("Shape:", rgb.shape)
print("Dtype:", rgb.dtype)


# -----------------------------
# 2. Convert tensor → NumPy
# -----------------------------

# CHW → HWC
rgb_np = rgb.permute(1, 2, 0).numpy()

print("\nConverted RGB:")
print("Shape:", rgb_np.shape)
print("Dtype:", rgb_np.dtype)


# -----------------------------
# 3. Load Depth Anything V2
# -----------------------------

MODEL_CONFIG = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    }
}

ENCODER = "vits"

CHECKPOINT = "models/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("\nUsing device:", DEVICE)

model = DepthAnythingV2(**MODEL_CONFIG[ENCODER])

model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))

model = model.to(DEVICE).eval()


# -----------------------------
# 4. Run depth inference
# -----------------------------

depth = model.infer_image(rgb_np, 518)


# -----------------------------
# 5. Inspect output
# -----------------------------

print("\nDepth output:")
print("Shape:", depth.shape)
print("Dtype:", depth.dtype)
print("Min:", depth.min())
print("Max:", depth.max())
print("Mean:", depth.mean())
