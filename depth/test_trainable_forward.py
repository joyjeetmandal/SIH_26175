import sys
import torch

sys.path.append(".")

sys.path.append("models/Depth-Anything-V2")

from depth_anything_v2.dpt import DepthAnythingV2
from datasets.gamus import GAMUSDataset


# -----------------------------
# 1. Load GAMUS
# -----------------------------

dataset = GAMUSDataset("data/Gamus")

sample = dataset[0]

rgb = sample["rgb"]

# Add batch dimension
rgb = rgb.unsqueeze(0)

print("Original RGB:")
print("Shape:", rgb.shape)
print("Dtype:", rgb.dtype)


# -----------------------------
# 2. Prepare RGB tensor
# -----------------------------

# uint8 [0, 255]
# → float32 [0, 1]

rgb = rgb.float() / 255.0


# Depth Anything V2 expects
# normalized RGB

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)

std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

rgb = (rgb - mean) / std


# -----------------------------
# 3. Resize
# -----------------------------

rgb = torch.nn.functional.interpolate(
    rgb,
    size=(518, 518),
    mode="bilinear",
    align_corners=False,
)

print("\nPrepared RGB:")
print("Shape:", rgb.shape)
print("Dtype:", rgb.dtype)


# -----------------------------
# 4. Load model
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

model = model.to(DEVICE)

rgb = rgb.to(DEVICE)


# -----------------------------
# 5. Forward pass
# -----------------------------

print("\nRunning forward pass...")

depth = model(rgb)


# -----------------------------
# 6. Inspect output
# -----------------------------

print("\nDepth output:")

print("Shape:", depth.shape)
print("Dtype:", depth.dtype)

print("Min:", depth.min().item())
print("Max:", depth.max().item())
print("Mean:", depth.mean().item())


# -----------------------------
# 7. Check gradient graph
# -----------------------------

print("\nGradient information:")

print("Requires grad:", depth.requires_grad)
print("Grad function:", depth.grad_fn)
