import os
import sys
import argparse
from contextlib import nullcontext

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
# BACKWARD-COMPATIBLE PUBLIC CONSTANTS
# ============================================================

MODEL_CONFIG = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    }
}

ENCODER = "vits"

CHECKPOINT = os.path.join(
    PROJECT_ROOT, "models", "depth_anything_v2_gamus_5004_best.pth"
)

# IMPORTANT:
# run_pipeline.py imports INPUT_SIZE from this module.
# Keep this symbol even though the new inference path uses the crop size
# stored inside the trained artifact whenever available.
INPUT_SIZE = 518

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

# Active metadata is populated by load_model() so the old API:
#     model, scale, shift = load_model()
#     pred = predict(model, image, scale, shift)
# continues to work unchanged.
_ACTIVE = {
    "transform": {"mode": "direct"},
    "mean": np.asarray(DEFAULT_NORMALIZATION["mean"], dtype=np.float32),
    "std": np.asarray(DEFAULT_NORMALIZATION["std"], dtype=np.float32),
    "crop_size": INPUT_SIZE,
    "overlap": 0.25,
    "rgb_divisor": 255.0,
    "tta": False,
}

# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def _safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def _decode_height(z, transform):
    z = z.float()
    mode = transform.get("mode", "direct")

    if mode == "log1p":
        limit = float(transform.get("log_inverse_numerical_limit", 30.0))
        return torch.expm1(z.clamp(max=limit))

    if mode == "normalized":
        return z * float(transform["scale_m"])

    if mode == "direct":
        return z

    raise RuntimeError(f"Unsupported saved target transform: {mode}")


# ============================================================
# TRAINED DEPTHWIZARD HEIGHT MODEL
# ============================================================

class DepthWizardHeightModel(torch.nn.Module):
    """
    Matches the GAMUS fine-tuned model rather than vanilla DA-V2 inference.
    """

    def __init__(self, model_config):
        super().__init__()
        self.model_config = dict(model_config)
        self.net = DepthAnythingV2(**self.model_config)

        # Match training/export: remove vanilla DA-V2's final activation layer.
        self.net.depth_head.scratch.output_conv2[3] = torch.nn.Identity()

    def forward(self, x):
        h, w = x.shape[-2:]

        pad_h = (-h) % 14
        pad_w = (-w) % 14
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        encoder = self.model_config["encoder"]

        features = self.net.pretrained.get_intermediate_layers(
            x_pad,
            self.net.intermediate_layer_idx[encoder],
            return_class_token=True,
        )

        logits = self.net.depth_head(
            features,
            x_pad.shape[-2] // 14,
            x_pad.shape[-1] // 14,
        )

        # Same positive-height parameterization used by the trained model.
        height_latent = F.softplus(logits.float())

        return height_latent[..., :h, :w]


# ============================================================
# LOAD MODEL — OLD API PRESERVED
# ============================================================

def load_model():
    """
    Backward-compatible return signature:
        model, scale, shift = load_model()

    `scale` and `shift` are retained only so the existing run_pipeline.py
    does not need to change. The new DepthWizard artifact uses its saved
    target transform instead.
    """
    global _ACTIVE

    print("=" * 60)
    print("LOADING GAMUS FINE-TUNED DEPTHWIZARD MODEL")
    print("=" * 60)

    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{CHECKPOINT}\n\n"
            "Put the trained GAMUS checkpoint at this location "
            "or change CHECKPOINT in depth/inference.py."
        )

    print("Checkpoint:", CHECKPOINT)
    print("Device:", DEVICE)

    artifact = _safe_torch_load(CHECKPOINT)

    if not isinstance(artifact, dict):
        raise RuntimeError(
            "Expected a DepthWizard checkpoint/artifact dictionary, "
            "not a bare vanilla DA-V2 state_dict."
        )

    if "model" not in artifact:
        raise RuntimeError(
            "Checkpoint does not contain a 'model' key. "
            f"Top-level keys: {list(artifact.keys())[:40]}"
        )

    model_config = artifact.get("model_config", MODEL_CONFIG[ENCODER])
    transform = artifact.get("transform", {"mode": "direct"})
    normalization = artifact.get("normalization", DEFAULT_NORMALIZATION)
    saved_config = artifact.get("config", {})

    state_dict = _strip_module_prefix(artifact["model"])

    # Training wrapper saves keys as net.pretrained..., net.depth_head...
    # If an export contains only the inner model, wrap those keys back.
    if state_dict and not any(k.startswith("net.") for k in list(state_dict.keys())[:30]):
        state_dict = {f"net.{k}": v for k, v in state_dict.items()}

    model = DepthWizardHeightModel(model_config)

    try:
        info = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        print("\nFirst checkpoint keys:")
        for key in list(state_dict.keys())[:20]:
            print(" ", key)
        raise RuntimeError(
            "Checkpoint/model architecture mismatch. "
            "This GAMUS model must be loaded through the DepthWizard height wrapper.\n"
            + str(exc)
        ) from exc

    count = sum(p.numel() for p in model.parameters())
    if not 22_000_000 < count < 28_000_000:
        raise RuntimeError(
            f"Unexpected model parameter count: {count:,}. "
            "Expected Depth Anything V2 Small."
        )

    _ACTIVE = {
        "transform": transform,
        "mean": np.asarray(normalization["mean"], dtype=np.float32),
        "std": np.asarray(normalization["std"], dtype=np.float32),
        "crop_size": int(artifact.get("crop_size", saved_config.get("CROP_SIZE", INPUT_SIZE))),
        "overlap": float(saved_config.get("TILE_OVERLAP", 0.25)),
        "rgb_divisor": float(saved_config.get("RGB_DIVISOR", 255.0)),
        "tta": bool(artifact.get("final_tta", False)),
    }

    model = model.to(DEVICE).eval()
    model.requires_grad_(False)

    print("Strict load:", info)
    print(f"Parameters: {count:,}")
    print("Saved target transform:", _ACTIVE["transform"])
    print("Saved tile/crop size:", _ACTIVE["crop_size"])
    print("Saved overlap:", _ACTIVE["overlap"])
    print("Saved TTA:", _ACTIVE["tta"])
    print("Model loaded successfully.\n")

    # Compatibility only. predict() intentionally does NOT use these
    # legacy scale/shift values for the new model.
    scale = 1.0
    shift = 0.0

    return model, scale, shift


# ============================================================
# IMAGE LOADING — OLD API PRESERVED
# ============================================================

def load_image(image_path):
    extension = os.path.splitext(image_path)[1].lower()

    if extension in [".tif", ".tiff"]:
        with rasterio.open(image_path) as src:
            print("Input CRS:", src.crs)
            print("Input transform:", src.transform)
            print("Input size:", src.width, "x", src.height)
            print("Input bands:", src.count)

            if src.count < 3:
                raise ValueError("GeoTIFF must contain at least 3 bands for RGB inference.")

            image = src.read([1, 2, 3])
            image = np.transpose(image, (1, 2, 0))
            metadata = src.meta.copy()

        return image, metadata

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image:\n{image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, None


# ============================================================
# PREPROCESS — OLD PUBLIC FUNCTION PRESERVED
# ============================================================

def preprocess(image):
    """
    Kept because existing code may import preprocess().

    For best/final predictions, use predict(), which performs the exact
    native-resolution tiled preprocessing used by validation/test.
    """
    rgb = image.astype(np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = np.clip(rgb / _ACTIVE["rgb_divisor"], 0.0, 1.0)
    rgb = (rgb - _ACTIVE["mean"]) / _ACTIVE["std"]

    x = torch.from_numpy(
        np.ascontiguousarray(rgb.transpose(2, 0, 1))
    ).unsqueeze(0)

    return x


# ============================================================
# TILED INFERENCE
# ============================================================

def _tile_starts(length, tile, stride):
    if length <= tile:
        return [0]
    return sorted(set(list(range(0, length - tile + 1, stride)) + [length - tile]))


def _amp_context():
    if DEVICE != "cuda":
        return nullcontext()

    major, _ = torch.cuda.get_device_capability(0)
    native_bf16 = major >= 8 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if native_bf16 else torch.float16

    return torch.amp.autocast("cuda", dtype=dtype)


@torch.inference_mode()
def _predict_native_tiled(model, image):
    transform = _ACTIVE["transform"]
    mean = _ACTIVE["mean"]
    std = _ACTIVE["std"]
    tile = int(_ACTIVE["crop_size"])
    overlap = float(_ACTIVE["overlap"])
    rgb_divisor = float(_ACTIVE["rgb_divisor"])
    tta = bool(_ACTIVE["tta"])

    if not (0 <= overlap < 0.75):
        raise ValueError("TILE_OVERLAP must be in [0, 0.75).")

    rgb = image.astype(np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)

    observed_max = float(np.max(rgb)) if rgb.size else 0.0
    if observed_max > rgb_divisor * 1.5:
        print(
            f"WARNING: input RGB max={observed_max:.1f} is much larger than "
            f"saved RGB_DIVISOR={rgb_divisor}. Check the image radiometric scale."
        )

    rgb = np.clip(rgb / rgb_divisor, 0.0, 1.0)

    h, w = rgb.shape[:2]
    stride = max(1, round(tile * (1.0 - overlap)))

    hann = np.maximum(np.hanning(tile).astype(np.float32), 0.025)
    weight = hann[:, None] * hann[None, :]

    output = np.zeros((h, w), dtype=np.float32)
    denominator = np.zeros((h, w), dtype=np.float32)

    operations = [(0, None)]
    if tta:
        operations = [
            (0, None),
            (0, -1),
            (0, -2),
            (1, None),
            (2, None),
            (3, None),
        ]

    for top in _tile_starts(h, tile, stride):
        for left in _tile_starts(w, tile, stride):
            patch = rgb[top:top + tile, left:left + tile]
            ph, pw = patch.shape[:2]

            patch = np.pad(
                patch,
                ((0, tile - ph), (0, tile - pw), (0, 0)),
                mode="edge",
            )

            patch = (patch - mean) / std

            x = torch.from_numpy(
                np.ascontiguousarray(patch.transpose(2, 0, 1))
            ).unsqueeze(0).to(DEVICE)

            prediction = torch.zeros(
                (1, 1, tile, tile),
                dtype=torch.float32,
                device=DEVICE,
            )

            for rotation, flip in operations:
                view = torch.rot90(x, rotation, (-2, -1))

                if flip is not None:
                    view = view.flip(flip)

                with _amp_context():
                    latent = model(view)

                height = _decode_height(latent, transform)

                if flip is not None:
                    height = height.flip(flip)

                height = torch.rot90(height, -rotation, (-2, -1))
                prediction += height / len(operations)

            arr = prediction[0, 0, :ph, :pw].cpu().numpy()

            output[top:top + ph, left:left + pw] += arr * weight[:ph, :pw]
            denominator[top:top + ph, left:left + pw] += weight[:ph, :pw]

    if not (denominator > 0).all():
        raise RuntimeError("Tiled inference left uncovered pixels.")

    return (output / denominator).astype(np.float32)


# ============================================================
# PREDICT — OLD API PRESERVED
# ============================================================

def predict(model, image, scale=1.0, shift=0.0):
    """
    Backward-compatible signature used by the existing run_pipeline.py.

    scale/shift are intentionally ignored for the new DepthWizard artifact,
    because its saved target transform is applied internally.
    """
    return _predict_native_tiled(model, image)


# ============================================================
# SAVE AGL — OLD API PRESERVED
# ============================================================

def save_agl(agl, output_path, metadata=None):
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
        dst.write(agl.astype(np.float32), 1)

    print("Saved AGL GeoTIFF:", output_path)


def save_numpy(agl, output_path):
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    np.save(output_path, agl.astype(np.float32))
    print("Saved NumPy output:", output_path)


# ============================================================
# STANDALONE MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run DepthWizard GAMUS DA-V2-S inference."
    )

    parser.add_argument("input", help="Input JPG, PNG or GeoTIFF")

    parser.add_argument(
        "-o",
        "--output",
        default="outputs/depth/predicted_agl.tif",
        help="Output AGL TIFF path",
    )

    parser.add_argument("--npy", default=None, help="Optional NumPy output path")

    args = parser.parse_args()

    model, scale, shift = load_model()

    image, metadata = load_image(args.input)

    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Running inference...")

    agl = predict(model, image, scale, shift)

    print("AGL shape:", agl.shape)
    print("AGL dtype:", agl.dtype)
    print("AGL min:", float(np.nanmin(agl)))
    print("AGL max:", float(np.nanmax(agl)))
    print("AGL mean:", float(np.nanmean(agl)))

    save_agl(agl, args.output, metadata)

    if args.npy is not None:
        save_numpy(agl, args.npy)

    print("Inference complete.")


if __name__ == "__main__":
    main()
