#!/usr/bin/env python3

"""
SIH 26175 - Automated Terrain Reconstruction Pipeline

Supported input:
    JPG
    JPEG
    PNG
    GeoTIFF
    ZIP containing Sentinel-2 B02/B03/B04 GeoTIFFs
    Folder containing Sentinel-2 B02/B03/B04 GeoTIFFs

Pipeline:

    INPUT
      ↓
    Automatic detection
      ↓
    RGB preparation
      ↓
    Depth Anything V2 + GAMUS-5004
      ↓
    Height / AGL prediction
      ↓
    If georeferenced:
        ↓
        SRTM download
        ↓
        SRTM alignment
        ↓
        Provisional DSM
      ↓
    Three.js-ready outputs

Current provisional DSM:

    DSM = SRTM + predicted_height

Negative values are currently clipped to 0 m.

IMPORTANT:
    The DSM is still PROVISIONAL.
    GAMUS target semantics and vertical datum need to be
    scientifically verified after the SIH demo.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject
from rasterio.enums import Resampling

import torch


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_ROOT / "models" / "Depth-Anything-V2"

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "depth_anything_v2_gamus_5004_best.pth"

SRTM_DIR = PROJECT_ROOT / "data" / "dem"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "runs"

MODEL_SIZE = 518

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_DSM_TO_ZERO = True

SRTM_BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"

MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


# ============================================================
# LOGGING
# ============================================================


def section(title: str):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def info(message: str):
    print(f"[INFO] {message}")


def success(message: str):
    print(f"[ OK ] {message}")


def warning(message: str):
    print(f"[WARN] {message}")


def fail(message: str):
    print(f"[ERROR] {message}")


# ============================================================
# FILE TYPE DETECTION
# ============================================================


def detect_input_type(path: Path) -> str:

    if path.is_dir():
        return "directory"

    suffix = path.suffix.lower()

    if suffix in {
        ".jpg",
        ".jpeg",
        ".png",
    }:
        return "image"

    if suffix in {
        ".tif",
        ".tiff",
    }:
        return "geotiff"

    if suffix == ".zip":
        return "zip"

    raise ValueError(
        f"Unsupported input:\n{path}\n\n"
        "Supported inputs:\n"
        "  JPG / JPEG / PNG\n"
        "  TIF / TIFF\n"
        "  ZIP\n"
        "  Directory"
    )


# ============================================================
# SENTINEL BAND DETECTION
# ============================================================


def detect_sentinel_band_number(text: str):
    """
    Detect B02/B03/B04 from a filename or metadata string.
    """

    text = text.upper()

    # Match B02, B03, B04 while avoiding B020 etc.
    match = re.search(
        r"(?<![A-Z0-9])B0([234])(?![0-9])",
        text,
    )

    if match:
        return f"B0{match.group(1)}"

    return None


def get_band_descriptions(src):

    descriptions = []

    for i in range(1, src.count + 1):
        description = src.descriptions[i - 1]

        if description is None:
            description = ""

        descriptions.append(str(description))

    return descriptions


def detect_sentinel_bands_from_metadata(src):

    mapping = {}

    # --------------------------------------------------------
    # Band descriptions
    # --------------------------------------------------------

    descriptions = get_band_descriptions(src)

    for band_number, description in enumerate(
        descriptions,
        start=1,
    ):
        detected = detect_sentinel_band_number(description)

        if detected:
            mapping[detected] = band_number

    # --------------------------------------------------------
    # Per-band tags
    # --------------------------------------------------------

    for band_number in range(
        1,
        src.count + 1,
    ):
        try:
            tags = src.tags(band_number)

            combined = " ".join(f"{key}={value}" for key, value in tags.items())

            detected = detect_sentinel_band_number(combined)

            if detected:
                mapping[detected] = band_number

        except Exception:
            pass

    if all(
        band in mapping
        for band in (
            "B02",
            "B03",
            "B04",
        )
    ):
        return mapping

    return None


# ============================================================
# PERCENTILE NORMALIZATION
# ============================================================


def percentile_stretch(
    channel: np.ndarray,
) -> np.ndarray:

    channel = channel.astype(np.float32)

    finite = np.isfinite(channel)

    if not np.any(finite):
        return np.zeros_like(
            channel,
            dtype=np.float32,
        )

    values = channel[finite]

    low = np.percentile(
        values,
        2,
    )

    high = np.percentile(
        values,
        98,
    )

    if high <= low:
        low = values.min()
        high = values.max()

        if high <= low:
            return np.zeros_like(
                channel,
                dtype=np.float32,
            )

    result = (channel - low) / (high - low)

    result = np.clip(
        result,
        0.0,
        1.0,
    )

    result[~finite] = 0.0

    return result.astype(np.float32)


def make_uint8_rgb(
    rgb: np.ndarray,
) -> np.ndarray:

    rgb = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    rgb = np.clip(
        rgb,
        0.0,
        1.0,
    )

    return (rgb * 255.0).round().astype(np.uint8)


# ============================================================
# SENTINEL BAND FILE VALIDATION
# ============================================================


def validate_sentinel_band_files(
    band_paths: dict,
):
    """
    Validate that B02/B03/B04 have matching spatial grids.
    """

    required = {
        "B02",
        "B03",
        "B04",
    }

    if set(band_paths.keys()) != required:
        missing = required - set(band_paths.keys())

        raise RuntimeError("Missing Sentinel bands: " + ", ".join(sorted(missing)))

    reference = None

    for band_name in (
        "B02",
        "B03",
        "B04",
    ):
        path = band_paths[band_name]

        with rasterio.open(path) as src:
            info(f"{band_name}: {src.width} × {src.height}, CRS={src.crs}")

            if reference is None:
                reference = {
                    "width": src.width,
                    "height": src.height,
                    "crs": src.crs,
                    "transform": src.transform,
                    "res": src.res,
                }

            else:
                if src.width != reference["width"]:
                    raise RuntimeError(f"{band_name} width does not match B02.")

                if src.height != reference["height"]:
                    raise RuntimeError(f"{band_name} height does not match B02.")

                if src.crs != reference["crs"]:
                    raise RuntimeError(f"{band_name} CRS does not match B02.")

                if not src.transform.almost_equals(reference["transform"]):
                    raise RuntimeError(f"{band_name} transform does not match B02.")


# ============================================================
# LOAD SENTINEL BANDS
# ============================================================


def load_sentinel_band_files(
    band_paths: dict,
):

    section("LOADING SENTINEL-2 B02/B03/B04")

    validate_sentinel_band_files(band_paths)

    with rasterio.open(band_paths["B04"]) as red_src:
        red = red_src.read(1).astype(np.float32)

        profile = red_src.profile.copy()

        crs = red_src.crs

        transform = red_src.transform

        bounds = red_src.bounds

        width = red_src.width

        height = red_src.height

        resolution = red_src.res

    with rasterio.open(band_paths["B03"]) as green_src:
        green = green_src.read(1).astype(np.float32)

    with rasterio.open(band_paths["B02"]) as blue_src:
        blue = blue_src.read(1).astype(np.float32)

    # --------------------------------------------------------
    # Sentinel reflectance
    # --------------------------------------------------------

    rgb = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=-1,
    )

    rgb_float = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Sentinel values are already reflectance-like.
    #
    # IMPORTANT:
    # Do NOT divide by 255.
    #
    rgb_float = np.clip(
        rgb_float,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Texture version
    # --------------------------------------------------------

    texture_rgb = make_uint8_rgb(
        np.stack(
            [
                percentile_stretch(red),
                percentile_stretch(green),
                percentile_stretch(blue),
            ],
            axis=-1,
        )
    )

    metadata = {
        "input_type": "sentinel_bands",
        "input_kind": "sentinel_rgb",
        "georeferenced": crs is not None,
        "width": int(width),
        "height": int(height),
        "bands": 3,
        "band_mapping": {
            "B04": "Red",
            "B03": "Green",
            "B02": "Blue",
        },
        "crs": str(crs) if crs else None,
        "transform": list(transform),
        "bounds": [
            float(bounds.left),
            float(bounds.bottom),
            float(bounds.right),
            float(bounds.top),
        ],
        "resolution": [
            float(resolution[0]),
            float(resolution[1]),
        ],
    }

    success("B02/B03/B04 successfully combined internally.")

    return (
        rgb_float,
        texture_rgb,
        metadata,
        profile,
    )


# ============================================================
# FIND SENTINEL FILES IN DIRECTORY
# ============================================================


def find_sentinel_bands_in_directory(
    directory: Path,
):

    band_paths = {}

    tif_files = list(directory.rglob("*.tif"))

    tif_files += list(directory.rglob("*.tiff"))

    if not tif_files:
        raise RuntimeError(f"No GeoTIFF files found in:\n{directory}")

    for path in tif_files:
        detected = detect_sentinel_band_number(path.name)

        if detected in {
            "B02",
            "B03",
            "B04",
        }:
            # First matching file wins.
            if detected not in band_paths:
                band_paths[detected] = path

    if all(
        band in band_paths
        for band in (
            "B02",
            "B03",
            "B04",
        )
    ):
        return band_paths

    return None


# ============================================================
# ZIP EXTRACTION
# ============================================================


def extract_sentinel_bands_from_zip(
    zip_path: Path,
    extraction_dir: Path,
):

    section("INSPECTING ZIP FILE")

    info(f"ZIP: {zip_path}")

    band_members = {}

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as archive:
        members = archive.infolist()

        for member in members:
            if member.is_dir():
                continue

            filename = Path(member.filename).name

            suffix = Path(filename).suffix.lower()

            if suffix not in {
                ".tif",
                ".tiff",
            }:
                continue

            detected = detect_sentinel_band_number(filename)

            if detected in {
                "B02",
                "B03",
                "B04",
            }:
                if detected not in band_members:
                    band_members[detected] = member

        if not all(
            band in band_members
            for band in (
                "B02",
                "B03",
                "B04",
            )
        ):
            found = ", ".join(sorted(band_members.keys())) if band_members else "none"

            raise RuntimeError(
                "The ZIP does not contain all required "
                "Sentinel bands B02, B03 and B04.\n"
                f"Detected: {found}\n\n"
                "Please download the Raw B02, B03 and B04 bands."
            )

        extracted = {}

        for band_name in (
            "B02",
            "B03",
            "B04",
        ):
            member = band_members[band_name]

            destination = extraction_dir / f"{band_name}.tiff"

            info(f"Extracting {band_name}...")

            with archive.open(member) as source:
                with open(
                    destination,
                    "wb",
                ) as target:
                    shutil.copyfileobj(
                        source,
                        target,
                    )

            extracted[band_name] = destination

    success("ZIP contains B02, B03 and B04.")

    return extracted


# ============================================================
# LOAD ZIP INPUT
# ============================================================


def load_zip_input(
    zip_path: Path,
    working_dir: Path,
):

    band_paths = extract_sentinel_bands_from_zip(
        zip_path,
        working_dir,
    )

    return load_sentinel_band_files(band_paths)


# ============================================================
# LOAD DIRECTORY INPUT
# ============================================================


def load_directory_input(
    directory: Path,
):

    section("INSPECTING INPUT DIRECTORY")

    band_paths = find_sentinel_bands_in_directory(directory)

    if band_paths is None:
        raise RuntimeError("Could not find B02, B03 and B04 GeoTIFFs in the directory.")

    for band_name, path in band_paths.items():
        info(f"{band_name}: {path}")

    return load_sentinel_band_files(band_paths)


# ============================================================
# LOAD NORMAL IMAGE
# ============================================================


def load_normal_image(
    path: Path,
):

    section("LOADING IMAGE")

    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise RuntimeError(f"Could not read image:\n{path}")

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    if image.ndim == 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB,
        )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    elif image.ndim == 3:
        channels = image.shape[2]

        if channels == 3:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

        elif channels == 4:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2RGB,
            )

        else:
            raise RuntimeError(f"Unsupported image channels: {channels}")

    else:
        raise RuntimeError(f"Unsupported image shape: {image.shape}")

    # --------------------------------------------------------
    # Convert to 0-1
    # --------------------------------------------------------

    if image.dtype == np.uint8:
        rgb_float = image.astype(np.float32) / 255.0

    elif image.dtype == np.uint16:
        rgb_float = image.astype(np.float32) / 65535.0

    else:
        rgb_float = percentile_stretch(image)

    rgb_float = np.clip(
        rgb_float,
        0.0,
        1.0,
    )

    texture_rgb = make_uint8_rgb(rgb_float)

    metadata = {
        "input_type": "image",
        "input_kind": "rgb_image",
        "georeferenced": False,
        "width": int(rgb_float.shape[1]),
        "height": int(rgb_float.shape[0]),
        "bands": 3,
    }

    success(f"Image size: {rgb_float.shape[1]} × {rgb_float.shape[0]}")

    return (
        rgb_float,
        texture_rgb,
        metadata,
        None,
    )


# ============================================================
# LOAD GEOTIFF
# ============================================================


def load_geotiff(
    path: Path,
):

    section("LOADING GEOTIFF")

    with rasterio.open(path) as src:
        if src.count < 3:
            raise RuntimeError(
                f"GeoTIFF has {src.count} band(s).\nAt least 3 bands are required."
            )

        # ----------------------------------------------------
        # Try Sentinel metadata
        # ----------------------------------------------------

        sentinel_mapping = detect_sentinel_bands_from_metadata(src)

        if sentinel_mapping is not None:
            info("Sentinel B02/B03/B04 detected from GeoTIFF metadata.")

            info(f"B02 = Band {sentinel_mapping['B02']}")

            info(f"B03 = Band {sentinel_mapping['B03']}")

            info(f"B04 = Band {sentinel_mapping['B04']}")

            red = src.read(sentinel_mapping["B04"]).astype(np.float32)

            green = src.read(sentinel_mapping["B03"]).astype(np.float32)

            blue = src.read(sentinel_mapping["B02"]).astype(np.float32)

            rgb = np.stack(
                [
                    red,
                    green,
                    blue,
                ],
                axis=-1,
            )

            rgb_float = np.nan_to_num(
                rgb,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            rgb_float = np.clip(
                rgb_float,
                0.0,
                1.0,
            )

            texture_rgb = make_uint8_rgb(
                np.stack(
                    [
                        percentile_stretch(red),
                        percentile_stretch(green),
                        percentile_stretch(blue),
                    ],
                    axis=-1,
                )
            )

            input_kind = "sentinel_rgb"

        else:
            warning("Could not identify Sentinel B02/B03/B04 metadata.")

            warning("Using first three bands as RGB.")

            band1 = src.read(1).astype(np.float32)

            band2 = src.read(2).astype(np.float32)

            band3 = src.read(3).astype(np.float32)

            rgb_float = np.stack(
                [
                    percentile_stretch(band1),
                    percentile_stretch(band2),
                    percentile_stretch(band3),
                ],
                axis=-1,
            )

            texture_rgb = make_uint8_rgb(rgb_float)

            input_kind = "generic_geotiff"

        profile = src.profile.copy()

        metadata = {
            "input_type": "geotiff",
            "input_kind": input_kind,
            "georeferenced": src.crs is not None,
            "width": int(src.width),
            "height": int(src.height),
            "bands": int(src.count),
            "crs": (str(src.crs) if src.crs else None),
            "transform": list(src.transform),
            "bounds": [
                float(src.bounds.left),
                float(src.bounds.bottom),
                float(src.bounds.right),
                float(src.bounds.top),
            ],
            "resolution": [
                float(src.res[0]),
                float(src.res[1]),
            ],
        }

    success(f"GeoTIFF size: {rgb_float.shape[1]} × {rgb_float.shape[0]}")

    return (
        rgb_float,
        texture_rgb,
        metadata,
        profile,
    )


# ============================================================
# MODEL LOADING
# ============================================================


def load_model():

    section("LOADING DEPTH MODEL")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Depth Anything V2 directory not found:\n{MODEL_DIR}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"GAMUS-5004 checkpoint not found:\n{CHECKPOINT_PATH}")

    sys.path.insert(
        0,
        str(MODEL_DIR),
    )

    from depth_anything_v2.dpt import (
        DepthAnythingV2,
    )

    info(f"Device: {DEVICE}")

    info(f"Checkpoint: {CHECKPOINT_PATH}")

    model = DepthAnythingV2(**MODEL_CONFIG)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(DEVICE)

    model.eval()

    learned_scale = float(
        checkpoint.get(
            "scale",
            1.0,
        )
        if isinstance(
            checkpoint,
            dict,
        )
        else 1.0
    )

    learned_shift = float(
        checkpoint.get(
            "shift",
            0.0,
        )
        if isinstance(
            checkpoint,
            dict,
        )
        else 0.0
    )

    info(f"Learned scale: {learned_scale}")

    info(f"Learned shift: {learned_shift}")

    success("Depth model loaded.")

    return (
        model,
        learned_scale,
        learned_shift,
    )


# ============================================================
# MODEL PREPROCESSING
# ============================================================


def prepare_model_input(
    rgb_float: np.ndarray,
):

    image = rgb_float.astype(np.float32)

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    image = cv2.resize(
        image,
        (
            MODEL_SIZE,
            MODEL_SIZE,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    image = np.transpose(
        image,
        (2, 0, 1),
    )

    tensor = torch.from_numpy(image).unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# DEPTH INFERENCE
# ============================================================


def run_depth_inference(
    model,
    rgb_float,
    learned_scale,
    learned_shift,
):

    section("RUNNING DEPTH INFERENCE")

    tensor = prepare_model_input(rgb_float)

    info(f"Model input: {tuple(tensor.shape)}")

    with torch.no_grad():
        predicted = model(tensor)

    predicted = predicted.squeeze().detach().cpu().numpy()

    info(f"Raw output shape: {predicted.shape}")

    output_height = rgb_float.shape[0]

    output_width = rgb_float.shape[1]

    predicted = cv2.resize(
        predicted.astype(np.float32),
        (
            output_width,
            output_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    predicted_height = learned_scale * predicted + learned_shift

    predicted_height = predicted_height.astype(np.float32)

    print()
    print("Predicted height statistics:")

    print(f"  Min:    {np.nanmin(predicted_height):.6f}")

    print(f"  Max:    {np.nanmax(predicted_height):.6f}")

    print(f"  Mean:   {np.nanmean(predicted_height):.6f}")

    print(f"  Median: {np.nanmedian(predicted_height):.6f}")

    success("Depth/height map generated.")

    return predicted_height


# ============================================================
# SRTM TILE NAMING
# ============================================================


def latitude_tile_name(
    lat: int,
):

    if lat >= 0:
        return f"N{lat:02d}"

    return f"S{abs(lat):02d}"


def longitude_tile_name(
    lon: int,
):

    if lon >= 0:
        return f"E{lon:03d}"

    return f"W{abs(lon):03d}"


def srtm_tile_name(
    lat: int,
    lon: int,
):

    return latitude_tile_name(lat) + longitude_tile_name(lon)


# ============================================================
# REQUIRED SRTM TILES
# ============================================================


def required_srtm_tiles(
    bounds,
):

    left, bottom, right, top = bounds

    eps = 1e-10

    min_lon = math.floor(left)

    max_lon = math.floor(right - eps)

    min_lat = math.floor(bottom)

    max_lat = math.floor(top - eps)

    tiles = []

    for lat in range(
        min_lat,
        max_lat + 1,
    ):
        for lon in range(
            min_lon,
            max_lon + 1,
        ):
            tiles.append(
                (
                    lat,
                    lon,
                )
            )

    return tiles


# ============================================================
# DOWNLOAD SRTM
# ============================================================


def download_srtm_tile(
    lat: int,
    lon: int,
):

    SRTM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tile_name = srtm_tile_name(
        lat,
        lon,
    )

    hgt_path = SRTM_DIR / f"{tile_name}.hgt"

    gz_path = SRTM_DIR / f"{tile_name}.hgt.gz"

    if hgt_path.exists():
        success(f"SRTM already available: {tile_name}.hgt")

        return hgt_path

    url = f"{SRTM_BASE_URL}/{latitude_tile_name(lat)}/{tile_name}.hgt.gz"

    info(f"Downloading SRTM: {tile_name}")

    try:
        urllib.request.urlretrieve(
            url,
            gz_path,
        )

    except Exception as exc:
        if gz_path.exists():
            gz_path.unlink()

        raise RuntimeError(
            f"Failed to download SRTM tile {tile_name}.\nURL: {url}\nReason: {exc}"
        )

    info("Extracting SRTM...")

    with gzip.open(
        gz_path,
        "rb",
    ) as source:
        with open(
            hgt_path,
            "wb",
        ) as destination:
            shutil.copyfileobj(
                source,
                destination,
            )

    gz_path.unlink()

    success(f"SRTM ready: {hgt_path.name}")

    return hgt_path


# ============================================================
# SRTM MOSAIC
# ============================================================


def build_srtm_mosaic(
    tile_paths,
):

    section("BUILDING SRTM MOSAIC")

    datasets = []

    try:
        for path in tile_paths:
            info(f"Reading {path.name}")

            datasets.append(rasterio.open(path))

        mosaic, transform = merge(
            datasets,
            nodata=-32768,
        )

        profile = datasets[0].profile.copy()

        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            count=1,
            dtype="int16",
            nodata=-32768,
        )

        result = mosaic[0]

    finally:
        for dataset in datasets:
            dataset.close()

    success("SRTM mosaic created.")

    return (
        result,
        transform,
        profile,
    )


# ============================================================
# ALIGN SRTM
# ============================================================


def align_srtm_to_input(
    srtm,
    srtm_transform,
    input_profile,
):

    section("ALIGNING SRTM TO INPUT GRID")

    destination = np.full(
        (
            input_profile["height"],
            input_profile["width"],
        ),
        np.nan,
        dtype=np.float32,
    )

    source = srtm.astype(np.float32)

    source[source <= -32768] = -32768

    reproject(
        source=source,
        destination=destination,
        src_transform=srtm_transform,
        src_crs="EPSG:4326",
        src_nodata=-32768,
        dst_transform=input_profile["transform"],
        dst_crs=input_profile["crs"],
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    success("SRTM aligned to input grid.")

    print()
    print("Aligned SRTM statistics:")

    print(f"  Min:    {np.nanmin(destination):.6f}")

    print(f"  Max:    {np.nanmax(destination):.6f}")

    print(f"  Mean:   {np.nanmean(destination):.6f}")

    print(f"  Median: {np.nanmedian(destination):.6f}")

    return destination


# ============================================================
# SAVE GEOTIFF
# ============================================================


def save_geotiff(
    path: Path,
    array: np.ndarray,
    profile: dict,
):

    profile = profile.copy()

    profile.update(
        dtype="float32",
        count=1,
        compress="deflate",
        predictor=2,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:
        dst.write(
            array.astype(np.float32),
            1,
        )


# ============================================================
# HEIGHTMAP PNG
# ============================================================


def create_heightmap_png(
    heightmap,
    output_path,
):

    finite = np.isfinite(heightmap)

    if not np.any(finite):
        raise RuntimeError("Heightmap contains no valid values.")

    minimum = float(np.nanmin(heightmap))

    maximum = float(np.nanmax(heightmap))

    if maximum <= minimum:
        normalized = np.zeros_like(
            heightmap,
            dtype=np.float32,
        )

    else:
        normalized = (heightmap - minimum) / (maximum - minimum)

        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

    normalized[~finite] = 0.0

    image16 = (normalized * 65535.0).round().astype(np.uint16)

    cv2.imwrite(
        str(output_path),
        image16,
    )

    return (
        minimum,
        maximum,
    )


# ============================================================
# PROVISIONAL DSM
# ============================================================


def generate_provisional_dsm(
    srtm,
    predicted_height,
):

    section("GENERATING PROVISIONAL DSM")

    if srtm.shape != predicted_height.shape:
        raise ValueError(
            f"SRTM shape {srtm.shape} "
            f"does not match height shape "
            f"{predicted_height.shape}"
        )

    dsm = (srtm + predicted_height).astype(np.float32)

    valid = np.isfinite(dsm)

    clipped_pixels = int(np.sum(valid & (dsm < 0)))

    if CLIP_DSM_TO_ZERO:
        dsm = np.maximum(
            dsm,
            0.0,
        )

        info(f"Clipped {clipped_pixels} negative pixels to 0 m.")

    success("Provisional DSM generated.")

    print()
    print("DSM statistics:")

    print(f"  Min:    {np.nanmin(dsm):.6f}")

    print(f"  Max:    {np.nanmax(dsm):.6f}")

    print(f"  Mean:   {np.nanmean(dsm):.6f}")

    print(f"  Median: {np.nanmedian(dsm):.6f}")

    return (
        dsm,
        clipped_pixels,
    )


# ============================================================
# SAVE METADATA
# ============================================================


def save_metadata(
    path: Path,
    metadata: dict,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )


# ============================================================
# SAVE TEXTURE
# ============================================================


def save_texture(
    texture_rgb,
    output_path,
):

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            texture_rgb,
            cv2.COLOR_RGB2BGR,
        ),
    )


# ============================================================
# MAIN PIPELINE
# ============================================================


def run_pipeline(
    input_path: Path,
):

    start_time = datetime.now()

    section("SIH 26175 TERRAIN PIPELINE")

    info(f"Input: {input_path}")

    info(f"Device: {DEVICE}")

    input_type = detect_input_type(input_path)

    info(f"Input type: {input_type}")

    # --------------------------------------------------------
    # Create run directory
    # --------------------------------------------------------

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir = OUTPUT_DIR / timestamp

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    info(f"Output directory: {run_dir}")

    # --------------------------------------------------------
    # Temporary workspace
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory(prefix="sih_pipeline_") as temp:
        temp_dir = Path(temp)

        # ----------------------------------------------------
        # Load input
        # ----------------------------------------------------

        if input_type == "image":
            (
                rgb_float,
                texture_rgb,
                metadata,
                raster_profile,
            ) = load_normal_image(input_path)

        elif input_type == "geotiff":
            (
                rgb_float,
                texture_rgb,
                metadata,
                raster_profile,
            ) = load_geotiff(input_path)

        elif input_type == "zip":
            (
                rgb_float,
                texture_rgb,
                metadata,
                raster_profile,
            ) = load_zip_input(
                input_path,
                temp_dir,
            )

        elif input_type == "directory":
            (
                rgb_float,
                texture_rgb,
                metadata,
                raster_profile,
            ) = load_directory_input(input_path)

        else:
            raise RuntimeError(f"Unhandled input type: {input_type}")

        # ----------------------------------------------------
        # Save texture
        # ----------------------------------------------------

        texture_path = run_dir / "texture.png"

        save_texture(
            texture_rgb,
            texture_path,
        )

        success(f"Texture saved: {texture_path}")

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        (
            model,
            learned_scale,
            learned_shift,
        ) = load_model()

        # ----------------------------------------------------
        # Run inference
        # ----------------------------------------------------

        predicted_height = run_depth_inference(
            model=model,
            rgb_float=rgb_float,
            learned_scale=learned_scale,
            learned_shift=learned_shift,
        )

        # ----------------------------------------------------
        # Save NPY
        # ----------------------------------------------------

        depth_npy_path = run_dir / "depth.npy"

        np.save(
            depth_npy_path,
            predicted_height,
        )

        success(f"Depth array saved: {depth_npy_path}")

        # ----------------------------------------------------
        # GEOSPATIAL PIPELINE
        # ----------------------------------------------------

        if metadata.get("georeferenced"):
            section("GEOSPATIAL PROCESSING")

            if raster_profile is None:
                raise RuntimeError("Missing raster profile.")

            if raster_profile.get("crs") is None:
                raise RuntimeError("GeoTIFF has no CRS.")

            info(f"CRS: {raster_profile['crs']}")

            info(f"Bounds: {metadata['bounds']}")

            # ------------------------------------------------
            # Save depth GeoTIFF
            # ------------------------------------------------

            depth_tif_path = run_dir / "depth.tif"

            save_geotiff(
                depth_tif_path,
                predicted_height,
                raster_profile,
            )

            success(f"Depth GeoTIFF saved: {depth_tif_path}")

            # ------------------------------------------------
            # Determine SRTM tiles
            # ------------------------------------------------

            required_tiles = required_srtm_tiles(metadata["bounds"])

            info(f"Required SRTM tiles: {len(required_tiles)}")

            tile_names = []

            for lat, lon in required_tiles:
                tile_name = srtm_tile_name(
                    lat,
                    lon,
                )

                tile_names.append(tile_name)

                download_srtm_tile(
                    lat,
                    lon,
                )

            # ------------------------------------------------
            # Build SRTM mosaic
            # ------------------------------------------------

            tile_paths = [SRTM_DIR / f"{name}.hgt" for name in tile_names]

            (
                srtm,
                srtm_transform,
                srtm_profile,
            ) = build_srtm_mosaic(tile_paths)

            # ------------------------------------------------
            # Align SRTM
            # ------------------------------------------------

            aligned_srtm = align_srtm_to_input(
                srtm=srtm,
                srtm_transform=srtm_transform,
                input_profile=raster_profile,
            )

            # ------------------------------------------------
            # Save aligned SRTM
            # ------------------------------------------------

            aligned_srtm_path = run_dir / "srtm_aligned.tif"

            save_geotiff(
                aligned_srtm_path,
                aligned_srtm,
                raster_profile,
            )

            success(f"Aligned SRTM saved: {aligned_srtm_path}")

            # ------------------------------------------------
            # Generate DSM
            # ------------------------------------------------

            (
                dsm,
                clipped_pixels,
            ) = generate_provisional_dsm(
                srtm=aligned_srtm,
                predicted_height=predicted_height,
            )

            dsm_path = run_dir / "dsm.tif"

            save_geotiff(
                dsm_path,
                dsm,
                raster_profile,
            )

            success(f"DSM saved: {dsm_path}")

            # ------------------------------------------------
            # Heightmap NPY
            # ------------------------------------------------

            heightmap_npy_path = run_dir / "heightmap.npy"

            np.save(
                heightmap_npy_path,
                dsm,
            )

            # ------------------------------------------------
            # Heightmap PNG
            # ------------------------------------------------

            heightmap_png_path = run_dir / "heightmap.png"

            (
                heightmap_min,
                heightmap_max,
            ) = create_heightmap_png(
                dsm,
                heightmap_png_path,
            )

            success(f"Heightmap saved: {heightmap_png_path}")

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata.update(
                {
                    "processing": {
                        "model": "Depth Anything V2",
                        "checkpoint": str(CHECKPOINT_PATH),
                        "checkpoint_name": CHECKPOINT_PATH.name,
                        "model_size": MODEL_SIZE,
                        "device": DEVICE,
                        "learned_scale": learned_scale,
                        "learned_shift": learned_shift,
                    },
                    "srtm": {
                        "tiles": tile_names,
                        "source": SRTM_BASE_URL,
                        "aligned_to_input_grid": True,
                    },
                    "dsm": {
                        "method": "SRTM + predicted height",
                        "provisional": True,
                        "negative_values_clipped": CLIP_DSM_TO_ZERO,
                        "pixels_clipped": clipped_pixels,
                        "min": float(np.nanmin(dsm)),
                        "max": float(np.nanmax(dsm)),
                        "mean": float(np.nanmean(dsm)),
                        "median": float(np.nanmedian(dsm)),
                    },
                    "threejs": {
                        "texture": "texture.png",
                        "heightmap": "heightmap.png",
                        "heightmap_npy": "heightmap.npy",
                        "width": int(dsm.shape[1]),
                        "height": int(dsm.shape[0]),
                        "elevation_min": heightmap_min,
                        "elevation_max": heightmap_max,
                    },
                }
            )

        # ----------------------------------------------------
        # NON-GEOSPATIAL INPUT
        # ----------------------------------------------------

        else:
            section("NON-GEOSPATIAL PROCESSING")

            warning("Input has no geographic reference.")

            warning("Output is relative/provisional.")

            heightmap_npy_path = run_dir / "heightmap.npy"

            np.save(
                heightmap_npy_path,
                predicted_height,
            )

            heightmap_png_path = run_dir / "heightmap.png"

            (
                heightmap_min,
                heightmap_max,
            ) = create_heightmap_png(
                predicted_height,
                heightmap_png_path,
            )

            success(f"Relative heightmap saved: {heightmap_png_path}")

            metadata.update(
                {
                    "processing": {
                        "model": "Depth Anything V2",
                        "checkpoint": str(CHECKPOINT_PATH),
                        "checkpoint_name": CHECKPOINT_PATH.name,
                        "model_size": MODEL_SIZE,
                        "device": DEVICE,
                        "learned_scale": learned_scale,
                        "learned_shift": learned_shift,
                    },
                    "dsm": {
                        "provisional": True,
                        "absolute_elevation": False,
                        "min": float(np.nanmin(predicted_height)),
                        "max": float(np.nanmax(predicted_height)),
                        "mean": float(np.nanmean(predicted_height)),
                        "median": float(np.nanmedian(predicted_height)),
                    },
                    "threejs": {
                        "texture": "texture.png",
                        "heightmap": "heightmap.png",
                        "heightmap_npy": "heightmap.npy",
                        "width": int(predicted_height.shape[1]),
                        "height": int(predicted_height.shape[0]),
                        "elevation_min": heightmap_min,
                        "elevation_max": heightmap_max,
                    },
                }
            )

        # ----------------------------------------------------
        # Final metadata
        # ----------------------------------------------------

        end_time = datetime.now()

        metadata["pipeline"] = {
            "started": start_time.isoformat(),
            "finished": end_time.isoformat(),
            "input_file": str(input_path.resolve()),
            "output_directory": str(run_dir.resolve()),
        }

        metadata_path = run_dir / "metadata.json"

        save_metadata(
            metadata_path,
            metadata,
        )

        success(f"Metadata saved: {metadata_path}")

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    section("PIPELINE COMPLETE")

    print()

    print("Input:")

    print(f"  {input_path}")

    print()

    print("Output directory:")

    print(f"  {run_dir}")

    print()

    print("Generated files:")

    for file in sorted(run_dir.iterdir()):
        print(f"  ✓ {file.name}")

    print()

    print("Ready for Three.js reconstruction.")

    print()

    return {
    "run_dir": str(run_dir),
    "heightmap": str(heightmap_png_path),
    "texture": str(texture_path),
    "metadata": str(metadata_path)
}


# ============================================================
# COMMAND LINE
# ============================================================


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=("SIH 26175 automated terrain reconstruction pipeline.")
    )

    parser.add_argument(
        "input",
        nargs="?",
        help=("Input JPG/PNG/GeoTIFF/ZIP or Sentinel band directory."),
    )

    return parser.parse_args()

def main():

    args = parse_arguments()

    try:
        if args.input:
            input_path = Path(args.input).expanduser().resolve()

        else:
            section("SIH 26175 TERRAIN PIPELINE")

            input_text = input("Enter path to JPG/PNG/GeoTIFF/ZIP/folder: ").strip()

            if not input_text:
                fail("No input supplied.")

                sys.exit(1)

            input_path = Path(input_text).expanduser().resolve()

        run_pipeline(input_path)

    except KeyboardInterrupt:
        print()

        warning("Pipeline interrupted.")

        sys.exit(130)

    except Exception as exc:
        print()

        fail(f"{type(exc).__name__}: {exc}")

        print()

        fail("Pipeline failed.")

        sys.exit(1)


if __name__ == "__main__":
    main()
