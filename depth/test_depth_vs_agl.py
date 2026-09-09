import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(".")
sys.path.append("models/Depth-Anything-V2")

from depth_anything_v2.dpt import DepthAnythingV2
from datasets.gamus import GAMUSDataset


# ==========================================
# 1. Load GAMUS sample
# ==========================================

dataset = GAMUSDataset("data/Gamus")

sample = dataset[0]

rgb = sample["rgb"].unsqueeze(0)
agl = sample["agl"].unsqueeze(0)
mask = sample["mask"].unsqueeze(0)

print("RGB shape:", rgb.shape)
print("AGL shape:", agl.shape)
print("Mask shape:", mask.shape)


# ==========================================
# 2. Prepare RGB
# ==========================================

rgb = rgb.float() / 255.0

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)

std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

rgb = (rgb - mean) / std

rgb = F.interpolate(
    rgb,
    size=(518, 518),
    mode="bilinear",
    align_corners=False,
)


# ==========================================
# 3. Load Depth Anything V2
# ==========================================

MODEL_CONFIG = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    }
}

model = DepthAnythingV2(**MODEL_CONFIG["vits"])

CHECKPOINT = "models/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"

model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))

model.eval()


# ==========================================
# 4. Generate relative depth
# ==========================================

with torch.no_grad():
    predicted_depth = model(rgb)

print("\nPredicted depth:")
print("Shape:", predicted_depth.shape)
print("Min:", predicted_depth.min().item())
print("Max:", predicted_depth.max().item())
print("Mean:", predicted_depth.mean().item())


# ==========================================
# 5. Resize prediction to AGL resolution
# ==========================================

predicted_depth = F.interpolate(
    predicted_depth.unsqueeze(1),
    size=agl.shape[-2:],
    mode="bilinear",
    align_corners=True,
).squeeze(1)

print("\nAfter resizing:")
print("Prediction shape:", predicted_depth.shape)
print("AGL shape:", agl.squeeze(1).shape)


# ==========================================
# 6. Apply valid-pixel mask
# ==========================================

pred = predicted_depth.squeeze(0)
target = agl.squeeze(0).squeeze(0)
valid = mask.squeeze(0).squeeze(0).bool()

pred_valid = pred[valid]
target_valid = target[valid]

print("\nValid pixels:", valid.sum().item())
print("Total pixels:", valid.numel())


# ==========================================
# 7. Statistics
# ==========================================

print("\nValid prediction statistics:")
print("Min:", pred_valid.min().item())
print("Max:", pred_valid.max().item())
print("Mean:", pred_valid.mean().item())

print("\nValid AGL statistics:")
print("Min:", target_valid.min().item())
print("Max:", target_valid.max().item())
print("Mean:", target_valid.mean().item())


# ==========================================
# 8. Correlation
# ==========================================

pred_centered = pred_valid - pred_valid.mean()
target_centered = target_valid - target_valid.mean()

correlation = (pred_centered * target_centered).sum() / (
    torch.sqrt((pred_centered**2).sum()) * torch.sqrt((target_centered**2).sum())
)

print("\nCorrelation:")
print(correlation.item())


# ==========================================
# 9. Simple scale + shift fitting
# ==========================================

# We fit:
#
# AGL ≈ a * predicted_depth + b
#
# using least squares.

x = pred_valid
y = target_valid

x_mean = x.mean()
y_mean = y.mean()

a = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()

b = y_mean - a * x_mean

print("\nScale + shift:")
print("a =", a.item())
print("b =", b.item())


# ==========================================
# 10. Calibrated prediction
# ==========================================

calibrated = a * pred_valid + b

mae = torch.mean(torch.abs(calibrated - target_valid))

rmse = torch.sqrt(torch.mean((calibrated - target_valid) ** 2))

print("\nAfter scale + shift calibration:")
print("MAE :", mae.item(), "meters")
print("RMSE:", rmse.item(), "meters")
