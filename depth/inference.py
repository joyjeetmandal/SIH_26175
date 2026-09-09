import os
import sys
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import rasterio


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEPTH_ANYTHING_ROOT = os.path.join(PROJECT_ROOT, "models", "Depth-Anything-V2")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DEPTH_ANYTHING_ROOT)

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    }
}

ENCODER = "vits"

# Change this if your downloaded checkpoint has a different name/location.
CHECKPOINT = os.path.join(
    PROJECT_ROOT, "models", "depth_anything_v2_gamus_5004_best.pth"
)

INPUT_SIZE = 518


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD MODEL
# ============================================================


def load_model():
    print("=" * 60)
    print("LOADING GAMUS FINE-TUNED MODEL")
    print("=" * 60)

    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{CHECKPOINT}\n\n"
            "Put the downloaded GAMUS checkpoint at this location "
            "or change CHECKPOINT in this file."
        )

    print("Checkpoint:", CHECKPOINT)
    print("Device:", DEVICE)

    model = DepthAnythingV2(**MODEL_CONFIG[ENCODER])

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")

    # --------------------------------------------------------
    # Extract model state dictionary
    # --------------------------------------------------------

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model" in checkpoint:
        state_dict = checkpoint["model"]

    else:
        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove "module." prefix if checkpoint was trained
    # using DataParallel.
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]

        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)

    # --------------------------------------------------------
    # Load learned scale and shift
    # --------------------------------------------------------

    scale = checkpoint.get("scale", 1.0)
    shift = checkpoint.get("shift", 0.0)

    if torch.is_tensor(scale):
        scale = scale.item()

    if torch.is_tensor(shift):
        shift = shift.item()

    print("Learned scale:", scale)
    print("Learned shift:", shift)

    model = model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")
    print()

    return model, float(scale), float(shift)


# ============================================================
# IMAGE LOADING
# ============================================================


def load_image(image_path):
    """
    Load JPG/PNG/TIFF.

    Returns:
        image_rgb : H x W x 3 uint8
        metadata  : Rasterio metadata if input is GeoTIFF
                    otherwise None
    """

    extension = os.path.splitext(image_path)[1].lower()

    # --------------------------------------------------------
    # GeoTIFF
    # --------------------------------------------------------

    if extension in [".tif", ".tiff"]:
        with rasterio.open(image_path) as src:
            print("Input CRS:", src.crs)
            print("Input transform:", src.transform)
            print("Input size:", src.width, "x", src.height)
            print("Input bands:", src.count)

            if src.count >= 3:
                # Read first three bands
                image = src.read([1, 2, 3])

                # CHW -> HWC
                image = np.transpose(image, (1, 2, 0))

            else:
                raise ValueError(
                    "GeoTIFF must contain at least 3 bands for RGB inference."
                )

            metadata = src.meta.copy()

        # Convert to uint8 if necessary
        if image.dtype != np.uint8:
            image = image.astype(np.float32)

            min_value = np.nanmin(image)
            max_value = np.nanmax(image)

            if max_value > min_value:
                image = (image - min_value) / (max_value - min_value) * 255.0

            image = np.clip(image, 0, 255).astype(np.uint8)

        return image, metadata

    # --------------------------------------------------------
    # JPG / PNG / other OpenCV-supported images
    # --------------------------------------------------------

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image:\n{image_path}")

    # OpenCV gives BGR.
    # Convert to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image, None


# ============================================================
# PREPROCESSING
# ============================================================


def preprocess(image):
    """
    Exactly matches the important preprocessing used
    during GAMUS training.

    Training:
        HWC -> CHW
        float
        /255 if necessary
        resize to 518x518
    """

    # HWC -> CHW
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()

    image_tensor = image_tensor.float()

    # Same normalization as training
    if image_tensor.max() > 1.5:
        image_tensor = image_tensor / 255.0

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Resize to training input size
    image_tensor = F.interpolate(
        image_tensor,
        size=(INPUT_SIZE, INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    return image_tensor


# ============================================================
# MODEL INFERENCE
# ============================================================


def predict(model, image, scale, shift):
    """
    Run GAMUS fine-tuned depth inference.

    Returns:
        AGL map at original image resolution.
    """

    original_height, original_width = image.shape[:2]

    input_tensor = preprocess(image)
    input_tensor = input_tensor.to(DEVICE)

    with torch.inference_mode():
        # Depth Anything V2 forward pass
        prediction = model(input_tensor)

        # Handle possible output formats
        if isinstance(prediction, dict):
            if "pred" in prediction:
                prediction = prediction["pred"]

            elif "depth" in prediction:
                prediction = prediction["depth"]

            else:
                raise RuntimeError(
                    "Model returned a dictionary but no "
                    "'pred' or 'depth' key was found."
                )

        elif isinstance(prediction, (tuple, list)):
            prediction = prediction[0]

        # Make sure output is 4D
        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(1)

        elif prediction.ndim == 2:
            prediction = prediction.unsqueeze(0).unsqueeze(0)

        # Resize prediction to original image size
        prediction = F.interpolate(
            prediction,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )

        prediction = prediction.squeeze()

        # Apply learned GAMUS scale and shift
        prediction = prediction * scale + shift

        prediction = prediction.cpu().numpy()

    return prediction.astype(np.float32)


# ============================================================
# SAVE AGL
# ============================================================


def save_agl(agl, output_path, metadata=None):
    """
    Save predicted AGL.

    If metadata is available from a GeoTIFF input,
    preserve its geospatial information.

    Otherwise save a normal single-band float32 TIFF.
    """

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    height, width = agl.shape

    if metadata is not None:
        profile = metadata.copy()

        profile.update(
            {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": "float32",
                "compress": "lzw",
            }
        )

    else:
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "compress": "lzw",
        }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(agl, 1)

    print("Saved AGL GeoTIFF:", output_path)


# ============================================================
# SAVE NUMPY
# ============================================================


def save_numpy(agl, output_path):
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    np.save(output_path, agl)

    print("Saved NumPy output:", output_path)


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run GAMUS fine-tuned Depth Anything V2 inference and generate an AGL map."
        )
    )

    parser.add_argument("input", help="Input JPG, PNG or GeoTIFF")

    parser.add_argument(
        "-o",
        "--output",
        default="outputs/depth/predicted_agl.tif",
        help="Output AGL GeoTIFF path",
    )

    parser.add_argument("--npy", default=None, help="Optional NumPy output path")

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("GAMUS DEPTH INFERENCE")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, scale, shift = load_model()

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    print("Loading image:")
    print(args.input)
    print()

    image, metadata = load_image(args.input)

    print("Image shape:", image.shape)

    print("Image dtype:", image.dtype)

    print()

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print("Running inference...")

    agl = predict(model, image, scale, shift)

    print()
    print("=" * 60)
    print("PREDICTION")
    print("=" * 60)

    print("AGL shape:", agl.shape)
    print("AGL dtype:", agl.dtype)
    print("AGL min:", float(np.nanmin(agl)))
    print("AGL max:", float(np.nanmax(agl)))
    print("AGL mean:", float(np.nanmean(agl)))

    # --------------------------------------------------------
    # Save GeoTIFF
    # --------------------------------------------------------

    save_agl(agl, args.output, metadata)

    # --------------------------------------------------------
    # Optional NumPy
    # --------------------------------------------------------

    if args.npy is not None:
        save_numpy(agl, args.npy)

    print()
    print("=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
