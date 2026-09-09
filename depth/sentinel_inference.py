import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

# ============================================================
# PATHS
# ============================================================

# Change this if your project is somewhere else.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Depth Anything V2 source code
DEPTH_ANYTHING_DIR = PROJECT_ROOT / "models" / "Depth-Anything-V2"

# Your downloaded Sentinel-2 files
SENTINEL_DIR = PROJECT_ROOT / "data" / "sentinel"

# Your trained 5004-sample checkpoint
CHECKPOINT = PROJECT_ROOT / "models" / "depth_anything_v2_gamus_5004_best.pth"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "depth"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Add Depth Anything V2 to Python path
sys.path.insert(
    0,
    str(DEPTH_ANYTHING_DIR),
)

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_SIZE = 518

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIG = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    }
}


# ============================================================
# SENTINEL FILES
# ============================================================

B02_PATH = next(SENTINEL_DIR.glob("*B02*Raw*.tiff"))

B03_PATH = next(SENTINEL_DIR.glob("*B03*Raw*.tiff"))

B04_PATH = next(SENTINEL_DIR.glob("*B04*Raw*.tiff"))


# ============================================================
# PRINT DEVICE
# ============================================================

print("=" * 70)
print("SENTINEL-2 DEPTH INFERENCE")
print("=" * 70)

print(
    "Device:",
    DEVICE,
)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Depth Anything V2...")

model = DepthAnythingV2(**MODEL_CONFIG["vits"])

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"\nCheckpoint not found:\n{CHECKPOINT}\n"
        "\nMake sure the 5004 checkpoint is located at:\n"
        "trained_models/depth_anything_v2_gamus_5004_best.pth"
    )


checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
)


# ------------------------------------------------------------
# Load trained model weights
# ------------------------------------------------------------

model.load_state_dict(checkpoint["model_state_dict"])


# ------------------------------------------------------------
# Load learned scale and shift
# ------------------------------------------------------------

scale = checkpoint.get(
    "scale",
    torch.tensor(1.0),
)

shift = checkpoint.get(
    "shift",
    torch.tensor(0.0),
)

scale = float(scale.item())

shift = float(shift.item())


print("Checkpoint loaded successfully.")

print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

print(f"Validation RMSE: {checkpoint.get('val_rmse', 'unknown')}")

print(f"Learned scale: {scale:.8f}")

print(f"Learned shift: {shift:.8f}")


# ------------------------------------------------------------
# Freeze encoder
# ------------------------------------------------------------

for param in model.pretrained.parameters():
    param.requires_grad = False


model.pretrained.eval()

model = model.to(DEVICE)

model.eval()


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device=DEVICE,
    dtype=torch.float32,
).view(
    1,
    3,
    1,
    1,
)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    device=DEVICE,
    dtype=torch.float32,
).view(
    1,
    3,
    1,
    1,
)


# ============================================================
# LOAD SENTINEL BANDS
# ============================================================


def load_band(path):

    print(f"\nLoading:\n  {path.name}")

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

        bounds = src.bounds

        width = src.width

        height = src.height

    print(f"  Shape: {data.shape}")

    print(f"  Min:   {np.nanmin(data):.6f}")

    print(f"  Max:   {np.nanmax(data):.6f}")

    print(f"  Mean:  {np.nanmean(data):.6f}")

    print(f"  CRS:   {crs}")

    return (
        data,
        profile,
        transform,
        crs,
        bounds,
        width,
        height,
    )


# ============================================================
# LOAD B02 / B03 / B04
# ============================================================

b02, profile, transform, crs, bounds, width, height = load_band(B02_PATH)

b03, _, _, _, _, _, _ = load_band(B03_PATH)

b04, _, _, _, _, _, _ = load_band(B04_PATH)


# ============================================================
# CHECK BAND CONSISTENCY
# ============================================================

if not (b02.shape == b03.shape == b04.shape):
    raise ValueError("B02, B03 and B04 do not have the same dimensions.")


# ============================================================
# CREATE RGB IMAGE
# ============================================================

print("\nCreating RGB image...")

# Sentinel bands:
#
# B02 = Blue
# B03 = Green
# B04 = Red
#
# Model expects:
#
# RGB = Red, Green, Blue

rgb = np.stack(
    [
        b04,
        b03,
        b02,
    ],
    axis=-1,
)


print(
    "RGB shape:",
    rgb.shape,
)

print(
    "RGB dtype:",
    rgb.dtype,
)

print(
    "RGB min:",
    np.nanmin(rgb),
)

print(
    "RGB max:",
    np.nanmax(rgb),
)


# ============================================================
# CHECK FOR INVALID VALUES
# ============================================================

if not np.isfinite(rgb).all():
    print("\nWARNING: RGB contains NaN or infinite values.")

    rgb = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


# ============================================================
# PREPARE RGB FOR MODEL
# ============================================================


def prepare_rgb(rgb):

    # --------------------------------------------------------
    # Convert NumPy → PyTorch
    # --------------------------------------------------------

    tensor = torch.from_numpy(rgb).float()

    # --------------------------------------------------------
    # HWC → BCHW
    # --------------------------------------------------------

    tensor = tensor.permute(
        2,
        0,
        1,
    )

    tensor = tensor.unsqueeze(0)

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # Sentinel-2 L2A raw bands downloaded here already contain
    # reflectance-like values around 0–1.
    #
    # Therefore DO NOT do:
    #
    # tensor = tensor / 255.0
    #
    # The training code did /255 because GAMUS images were
    # supplied as 0–255 image values.
    #
    # Sentinel reflectance is already scaled to approximately
    # 0–1.
    # --------------------------------------------------------

    tensor = tensor.to(
        DEVICE,
        non_blocking=True,
    )

    # --------------------------------------------------------
    # ImageNet normalization
    # --------------------------------------------------------

    tensor = (tensor - mean) / std

    # --------------------------------------------------------
    # Resize to Depth Anything input
    # --------------------------------------------------------

    tensor = F.interpolate(
        tensor,
        size=(
            MODEL_SIZE,
            MODEL_SIZE,
        ),
        mode="bilinear",
        align_corners=False,
    )

    return tensor


# ============================================================
# PREPARE INPUT
# ============================================================

input_tensor = prepare_rgb(rgb)


print(
    "\nModel input shape:",
    tuple(input_tensor.shape),
)

print(
    "Model input dtype:",
    input_tensor.dtype,
)

print(
    "Model input device:",
    input_tensor.device,
)


# ============================================================
# RUN MODEL
# ============================================================

print("\nRunning depth model...")

with torch.no_grad():
    predicted_depth = model(input_tensor)


# ============================================================
# MODEL OUTPUT
# ============================================================

print(
    "Raw model output shape:",
    tuple(predicted_depth.shape),
)


# ============================================================
# RESIZE DEPTH BACK TO ORIGINAL SENTINEL SIZE
# ============================================================

predicted_depth = F.interpolate(
    predicted_depth.unsqueeze(1),
    size=(
        height,
        width,
    ),
    mode="bilinear",
    align_corners=True,
).squeeze(1)


# Remove batch dimension

predicted_depth = predicted_depth.squeeze(0)


# Move to CPU

predicted_depth = predicted_depth.detach().cpu().numpy().astype(np.float32)


# ============================================================
# APPLY LEARNED SCALE + SHIFT
# ============================================================

predicted_agl = scale * predicted_depth + shift


predicted_agl = predicted_agl.astype(np.float32)


# ============================================================
# STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("DEPTH RESULTS")
print("=" * 70)

print(
    "Shape:",
    predicted_agl.shape,
)

print(
    "Dtype:",
    predicted_agl.dtype,
)

print(
    "Min:",
    np.nanmin(predicted_agl),
)

print(
    "Max:",
    np.nanmax(predicted_agl),
)

print(
    "Mean:",
    np.nanmean(predicted_agl),
)

print(
    "Median:",
    np.nanmedian(predicted_agl),
)

print("=" * 70)


# ============================================================
# SAVE NUMPY DEPTH
# ============================================================

npy_path = OUTPUT_DIR / "sentinel_depth_5004.npy"

np.save(
    npy_path,
    predicted_agl,
)


print("\nNumPy depth saved to:")

print(npy_path)


# ============================================================
# SAVE GEOTIFF
# ============================================================

geotiff_path = OUTPUT_DIR / "sentinel_depth_5004.tif"


output_profile = profile.copy()

output_profile.update(
    {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
)


with rasterio.open(
    geotiff_path,
    "w",
    **output_profile,
) as dst:
    dst.write(
        predicted_agl,
        1,
    )


print("\nGeoTIFF depth saved to:")

print(geotiff_path)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("INFERENCE COMPLETE")
print("=" * 70)

print("Input:")

print(f"  B02: {B02_PATH.name}")

print(f"  B03: {B03_PATH.name}")

print(f"  B04: {B04_PATH.name}")

print("\nModel:")

print(f"  {CHECKPOINT.name}")

print("\nOutput:")

print(f"  {geotiff_path}")

print(f"  {npy_path}")

print(
    "\nCRS:",
    crs,
)

print(
    "Resolution:",
    transform.a,
    "x",
    abs(transform.e),
)

print("=" * 70)
