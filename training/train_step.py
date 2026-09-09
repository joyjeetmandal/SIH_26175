import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(".")
sys.path.append("models/Depth-Anything-V2")

from depth_anything_v2.dpt import DepthAnythingV2
from datasets.gamus import GAMUSDataset


# ==========================================
# Configuration
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 10
BATCH_SIZE = 2

MODEL_CONFIG = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    }
}

CHECKPOINT = "models/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"


# ==========================================
# 1. Load dataset
# ==========================================

dataset = GAMUSDataset("data/Gamus")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

print("Dataset size:", len(dataset))
print("Batch size:", BATCH_SIZE)
print("Number of batches:", len(loader))


# ==========================================
# 2. Load Depth Anything V2
# ==========================================

model = DepthAnythingV2(**MODEL_CONFIG["vits"])

model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))

model = model.to(DEVICE)


# ==========================================
# 3. Freeze DINOv2 encoder
# ==========================================

for param in model.pretrained.parameters():
    param.requires_grad = False


model.train()

# Keep frozen encoder in evaluation mode
model.pretrained.eval()


# ==========================================
# 4. Learnable scale + shift
# ==========================================

scale = torch.nn.Parameter(torch.tensor(1.0, device=DEVICE))

shift = torch.nn.Parameter(torch.tensor(0.0, device=DEVICE))


# ==========================================
# 5. Optimizer
# ==========================================

optimizer = torch.optim.Adam(
    list(model.depth_head.parameters()) + [scale, shift],
    lr=1e-4,
)


# ==========================================
# 6. Parameter information
# ==========================================

total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nModel parameters:")
print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)


# ==========================================
# 7. Training
# ==========================================

print("\nStarting training...\n")


for epoch in range(EPOCHS):
    epoch_loss = 0.0

    # --------------------------------------
    # Loop through batches
    # --------------------------------------

    for batch_idx, batch in enumerate(loader):
        rgb = batch["rgb"]
        agl = batch["agl"]
        mask = batch["mask"]

        # ----------------------------------
        # Prepare RGB
        # ----------------------------------

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

        # ----------------------------------
        # Move to device
        # ----------------------------------

        rgb = rgb.to(DEVICE)
        agl = agl.to(DEVICE)
        mask = mask.to(DEVICE)

        # ----------------------------------
        # Forward pass
        # ----------------------------------

        predicted_depth = model(rgb)

        # ----------------------------------
        # Resize prediction to AGL size
        # ----------------------------------

        predicted_depth = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=agl.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)

        # ----------------------------------
        # Scale + shift
        # ----------------------------------

        predicted_agl = scale * predicted_depth + shift

        # ----------------------------------
        # Mask invalid pixels
        # ----------------------------------

        valid_mask = mask.squeeze(1).bool()

        pred_valid = predicted_agl[valid_mask]

        target_valid = agl.squeeze(1)[valid_mask]

        # ----------------------------------
        # Calculate loss
        # ----------------------------------

        loss = torch.mean((pred_valid - target_valid) ** 2)

        # ----------------------------------
        # Backpropagation
        # ----------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ----------------------------------
        # Accumulate loss
        # ----------------------------------

        epoch_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Batch [{batch_idx + 1}/{len(loader)}] "
            f"Loss: {loss.item():.4f}"
        )

    # --------------------------------------
    # Average epoch loss
    # --------------------------------------

    average_loss = epoch_loss / len(loader)

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}] Average Loss: {average_loss:.4f}")

    print(f"Scale: {scale.item():.6f} Shift: {shift.item():.6f}\n")


print("Training complete.")
