import gzip
import json
import math
import os
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
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling


# ============================================================
# SIH 26175 - END-TO-END PIPELINE
#
# Usage:
#   python run_pipeline.py "path/to/Browser_images.zip"
#
# This runner:
#   - really uses sys.argv[1]
#   - reads Sentinel B04/B03/B02 from ZIP
#   - preserves georeferencing
#   - runs depth/inference.py
#   - downloads SRTM from AWS Skadi
#   - aligns SRTM to the prediction grid
#   - runs dsm/generate.py
#   - creates heightmap.npy/png + texture.png + metadata.json
#   - launches the 3D visualizer on THIS exact run
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

try:
    from depth.inference import (
        CHECKPOINT,
        DEVICE,
        INPUT_SIZE,
        load_image,
        load_model,
        predict,
        save_agl,
        save_numpy,
    )
except Exception as exc:
    raise RuntimeError(
        "Could not import depth/inference.py. "
        "Expected it at: depth/inference.py"
    ) from exc

try:
    from dsm.generate import generate_dsm
except Exception as exc:
    raise RuntimeError(
        "Could not import dsm/generate.py. "
        "Expected it at: dsm/generate.py"
    ) from exc


def import_visualizer():
    try:
        from Reconstruction.visualize_3d import launch_3d_simulation_mesh
        return launch_3d_simulation_mesh
    except ModuleNotFoundError:
        try:
            from reconstruction.visualize_3d import launch_3d_simulation_mesh
            return launch_3d_simulation_mesh
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Could not find Reconstruction/visualize_3d.py "
                "or reconstruction/visualize_3d.py"
            ) from exc


# ============================================================
# CLI
# ============================================================

def get_input_path() -> Path:
    if len(sys.argv) != 2:
        print("Usage:")
        print('  python run_pipeline.py "path/to/JPG/PNG/GeoTIFF/ZIP"')
        sys.exit(2)

    p = Path(sys.argv[1]).expanduser()

    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()

    if not p.exists():
        print(f"\n❌ Input does not exist:\n   {p}")
        sys.exit(2)

    return p


def make_run_dir() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / stamp

    # Avoid accidental collision if launched twice within one second.
    counter = 1
    while run_dir.exists():
        run_dir = RUNS_DIR / f"{stamp}_{counter:02d}"
        counter += 1

    run_dir.mkdir(parents=True)
    return run_dir


# ============================================================
# SENTINEL ZIP HANDLING
# ============================================================

BAND_EXTENSIONS = {".tif", ".tiff", ".jp2"}


def _band_match(filename: str, band: str) -> bool:
    name = Path(filename).name.upper()
    suffix = Path(filename).suffix.lower()

    if suffix not in BAND_EXTENSIONS:
        return False

    # Matches names such as:
    # B04.tif, *_B04.tiff, *_B04_10m.jp2, etc.
    return re.search(rf"(^|[^A-Z0-9]){band}([^A-Z0-9]|$)", name) is not None


def find_band_member(names, band):
    matches = [n for n in names if _band_match(n, band)]

    if not matches:
        # Slightly looser fallback for unusual exported names.
        matches = [
            n for n in names
            if band in Path(n).name.upper()
            and Path(n).suffix.lower() in BAND_EXTENSIONS
        ]

    if not matches:
        raise FileNotFoundError(
            f"Could not find Sentinel {band} inside the ZIP."
        )

    # Prefer TIFF over JP2; then shortest path/name.
    matches.sort(
        key=lambda n: (
            0 if Path(n).suffix.lower() in {".tif", ".tiff"} else 1,
            len(n),
            n.lower(),
        )
    )
    return matches[0]


def extract_sentinel_bands(zip_path: Path, work_dir: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]

        selected = {
            "B04": find_band_member(names, "B04"),
            "B03": find_band_member(names, "B03"),
            "B02": find_band_member(names, "B02"),
        }

        print("\nSentinel bands found:")
        for band, member in selected.items():
            print(f"  {band}: {member}")

        result = {}

        for band, member in selected.items():
            suffix = Path(member).suffix.lower()
            out = work_dir / f"{band}{suffix}"

            with zf.open(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)

            result[band] = out

    return result


def read_band_on_reference(path: Path, ref_profile: dict, ref_crs, ref_transform,
                           ref_width: int, ref_height: int):
    with rasterio.open(path) as src:
        src_data = src.read(1).astype(np.float32)

        same_grid = (
            src.width == ref_width
            and src.height == ref_height
            and src.crs == ref_crs
            and src.transform == ref_transform
        )

        if same_grid:
            return src_data

        if src.crs is None or ref_crs is None:
            raise ValueError(
                f"Band grids differ, but CRS is missing for {path.name}."
            )

        dst = np.full((ref_height, ref_width), np.nan, dtype=np.float32)

        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        return dst


def build_sentinel_rgb_geotiff(bands: dict, output_path: Path):
    # B04 is the reference grid.
    with rasterio.open(bands["B04"]) as ref:
        if ref.crs is None:
            raise ValueError(
                "B04 has no CRS. Sentinel ZIP must contain georeferenced bands."
            )

        ref_profile = ref.profile.copy()
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_bounds = ref.bounds
        ref_res = ref.res

    red = read_band_on_reference(
        bands["B04"], ref_profile, ref_crs, ref_transform,
        ref_width, ref_height
    )
    green = read_band_on_reference(
        bands["B03"], ref_profile, ref_crs, ref_transform,
        ref_width, ref_height
    )
    blue = read_band_on_reference(
        bands["B02"], ref_profile, ref_crs, ref_transform,
        ref_width, ref_height
    )

    rgb = np.stack([red, green, blue], axis=0).astype(np.float32)

    profile = ref_profile.copy()
    profile.update(
        driver="GTiff",
        width=ref_width,
        height=ref_height,
        count=3,
        dtype="float32",
        compress="lzw",
        nodata=np.nan,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(rgb)

    return {
        "width": ref_width,
        "height": ref_height,
        "crs": ref_crs,
        "transform": ref_transform,
        "bounds": ref_bounds,
        "resolution": ref_res,
    }


# ============================================================
# SRTM
# ============================================================

SRTM_BASE = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"


def tile_name(lat_deg: int, lon_deg: int) -> str:
    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"

    return f"{ns}{abs(lat_deg):02d}{ew}{abs(lon_deg):03d}"


def required_srtm_tiles(bounds):
    left, bottom, right, top = bounds

    # Small epsilon avoids requesting the next tile when an edge lands
    # exactly on an integer degree boundary.
    eps = 1e-10

    lat_start = math.floor(bottom)
    lat_end = math.floor(top - eps)

    lon_start = math.floor(left)
    lon_end = math.floor(right - eps)

    tiles = []

    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            tiles.append((lat, lon, tile_name(lat, lon)))

    return tiles


def download_srtm_tile(lat: int, lon: int, name: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)

    tif_path = cache_dir / f"{name}.tif"

    if tif_path.exists():
        print(f"  SRTM cache: {name}")
        return tif_path

    region = name[:3]  # e.g. N22
    url = f"{SRTM_BASE}/{region}/{name}.hgt.gz"

    gz_path = cache_dir / f"{name}.hgt.gz"
    hgt_path = cache_dir / f"{name}.hgt"

    print(f"  Downloading SRTM: {name}")

    try:
        urllib.request.urlretrieve(url, gz_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download SRTM tile {name}\nURL: {url}\n{exc}"
        ) from exc

    with gzip.open(gz_path, "rb") as src, open(hgt_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    byte_size = hgt_path.stat().st_size

    if byte_size % 2 != 0:
        raise RuntimeError(f"Invalid HGT file size for {name}: {byte_size}")

    sample_count = byte_size // 2
    side = int(round(math.sqrt(sample_count)))

    if side * side != sample_count:
        raise RuntimeError(
            f"Could not infer square HGT dimensions for {name}. "
            f"Samples={sample_count}"
        )

    data = np.fromfile(hgt_path, dtype=">i2").reshape(side, side).astype(np.float32)

    # SRTM void value.
    data[data <= -32768] = np.nan

    pixel_size = 1.0 / (side - 1)
    transform = from_origin(
        lon,
        lat + 1,
        pixel_size,
        pixel_size,
    )

    profile = {
        "driver": "GTiff",
        "height": side,
        "width": side,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": np.nan,
        "compress": "lzw",
    }

    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(data, 1)

    # Keep only the GeoTIFF cache.
    for p in (gz_path, hgt_path):
        try:
            p.unlink()
        except OSError:
            pass

    return tif_path


def build_srtm_mosaic(bounds, output_path: Path):
    tiles = required_srtm_tiles(bounds)

    if not tiles:
        raise RuntimeError("No SRTM tiles determined from the input bounds.")

    cache_dir = PROJECT_ROOT / "data" / "srtm_cache"

    tile_paths = [
        download_srtm_tile(lat, lon, name, cache_dir)
        for lat, lon, name in tiles
    ]

    srcs = [rasterio.open(p) for p in tile_paths]

    try:
        mosaic, transform = merge(srcs, bounds=bounds)
        profile = srcs[0].profile.copy()

        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            count=1,
            dtype="float32",
            nodata=np.nan,
            compress="lzw",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic.astype(np.float32))

    finally:
        for src in srcs:
            src.close()

    return [name for _, _, name in tiles]


def align_dem_to_agl(dem_path: Path, agl_path: Path, output_path: Path):
    with rasterio.open(agl_path) as agl_src:
        if agl_src.crs is None:
            raise ValueError("Predicted AGL GeoTIFF has no CRS.")

        profile = agl_src.profile.copy()
        destination = np.full(
            (agl_src.height, agl_src.width),
            np.nan,
            dtype=np.float32,
        )

        with rasterio.open(dem_path) as dem_src:
            reproject(
                source=rasterio.band(dem_src, 1),
                destination=destination,
                src_transform=dem_src.transform,
                src_crs=dem_src.crs,
                src_nodata=dem_src.nodata,
                dst_transform=agl_src.transform,
                dst_crs=agl_src.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=np.nan,
            compress="lzw",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(destination, 1)


# ============================================================
# OUTPUT HELPERS
# ============================================================

def save_texture(image_rgb: np.ndarray, output_path: Path):
    if image_rgb.dtype != np.uint8:
        arr = image_rgb.astype(np.float32)
        finite = np.isfinite(arr)

        if finite.any():
            lo = float(np.nanpercentile(arr, 2))
            hi = float(np.nanpercentile(arr, 98))

            if hi > lo:
                arr = (arr - lo) / (hi - lo) * 255.0

        image_rgb = np.clip(arr, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if not cv2.imwrite(str(output_path), bgr):
        raise RuntimeError(f"Could not write texture: {output_path}")


def save_heightmap_png(heightmap: np.ndarray, output_path: Path):
    valid = np.isfinite(heightmap)

    if not valid.any():
        raise ValueError("Heightmap contains no finite pixels.")

    vmin = float(np.nanmin(heightmap))
    vmax = float(np.nanmax(heightmap))

    norm = np.zeros(heightmap.shape, dtype=np.uint16)

    if vmax > vmin:
        scaled = (heightmap - vmin) / (vmax - vmin)
        scaled = np.clip(scaled, 0.0, 1.0)
        norm[valid] = (scaled[valid] * 65535.0).astype(np.uint16)

    if not cv2.imwrite(str(output_path), norm):
        raise RuntimeError(f"Could not write heightmap PNG: {output_path}")


def clip_negative_dsm(dsm_path: Path):
    """
    Match the behavior shown in the project's previous successful
    metadata: negative DSM values are clipped to 0 m.
    """
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    finite = np.isfinite(dsm)
    negative = finite & (dsm < 0)
    count = int(negative.sum())

    if count:
        dsm[negative] = 0.0

        with rasterio.open(dsm_path, "w", **profile) as dst:
            dst.write(dsm, 1)

    return dsm, count


# ============================================================
# SENTINEL PIPELINE
# ============================================================

def process_sentinel_zip(input_path: Path, run_dir: Path, started: str):
    print("\n=========================================================================")
    print("SENTINEL ZIP PIPELINE")
    print("=========================================================================")

    with tempfile.TemporaryDirectory(prefix="sih_sentinel_") as tmp:
        work_dir = Path(tmp)

        bands = extract_sentinel_bands(input_path, work_dir)

        rgb_tif = run_dir / "sentinel_rgb.tif"

        geo = build_sentinel_rgb_geotiff(
            bands,
            rgb_tif,
        )

    print(f"\nRGB GeoTIFF: {rgb_tif}")
    print(f"Size: {geo['width']} x {geo['height']}")
    print(f"CRS: {geo['crs']}")
    print(f"Bounds: {geo['bounds']}")

    # --------------------------------------------------------
    # DEPTH / AGL INFERENCE - uses depth/inference.py
    # --------------------------------------------------------

    print("\n=========================================================================")
    print("DEPTH ANYTHING V2 / GAMUS INFERENCE")
    print("=========================================================================")

    model, scale, shift = load_model()

    image_rgb, raster_metadata = load_image(str(rgb_tif))

    agl = predict(
        model,
        image_rgb,
        scale,
        shift,
    )

    depth_tif = run_dir / "depth.tif"
    depth_npy = run_dir / "depth.npy"

    save_agl(
        agl,
        str(depth_tif),
        raster_metadata,
    )

    save_numpy(
        agl,
        str(depth_npy),
    )

    save_texture(
        image_rgb,
        run_dir / "texture.png",
    )

    # --------------------------------------------------------
    # SRTM
    # --------------------------------------------------------

    print("\n=========================================================================")
    print("SRTM")
    print("=========================================================================")

    # Geo bounds are in the Sentinel CRS. The successful example is
    # EPSG:4326. Reproject bounds if required.
    if str(geo["crs"]).upper() == "EPSG:4326":
        lonlat_bounds = tuple(geo["bounds"])
    else:
        from rasterio.warp import transform_bounds

        lonlat_bounds = transform_bounds(
            geo["crs"],
            "EPSG:4326",
            *geo["bounds"],
            densify_pts=21,
        )

    srtm_raw = run_dir / "srtm_raw.tif"

    srtm_tiles = build_srtm_mosaic(
        lonlat_bounds,
        srtm_raw,
    )

    srtm_aligned = run_dir / "srtm_aligned.tif"

    align_dem_to_agl(
        srtm_raw,
        depth_tif,
        srtm_aligned,
    )

    print(f"[ OK ] Aligned SRTM saved: {srtm_aligned}")

    # --------------------------------------------------------
    # DSM - uses dsm/generate.py
    # --------------------------------------------------------

    print("\n=========================================================================")
    print("DSM")
    print("=========================================================================")

    dsm_tif = run_dir / "dsm.tif"

    generate_dsm(
        agl_path=str(depth_tif),
        dem_path=str(srtm_aligned),
        output_path=str(dsm_tif),
    )

    # Previous successful project metadata explicitly recorded
    # negative_values_clipped=true, so keep that pipeline behavior.
    dsm, pixels_clipped = clip_negative_dsm(dsm_tif)

    valid = np.isfinite(dsm)

    if not valid.any():
        raise RuntimeError("Generated DSM contains no valid pixels.")

    dsm_min = float(np.nanmin(dsm))
    dsm_max = float(np.nanmax(dsm))
    dsm_mean = float(np.nanmean(dsm))
    dsm_median = float(np.nanmedian(dsm))

    # --------------------------------------------------------
    # HEIGHTMAP
    # --------------------------------------------------------

    heightmap_npy = run_dir / "heightmap.npy"
    heightmap_png = run_dir / "heightmap.png"

    np.save(heightmap_npy, dsm.astype(np.float32))

    save_heightmap_png(
        dsm,
        heightmap_png,
    )

    # The raw SRTM mosaic is internal; keep the aligned one.
    try:
        srtm_raw.unlink()
    except OSError:
        pass

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    transform = geo["transform"]
    bounds = geo["bounds"]

    finished = datetime.now().isoformat()

    metadata = {
        "input_type": "sentinel_bands",
        "input_kind": "sentinel_rgb",
        "georeferenced": True,
        "width": int(geo["width"]),
        "height": int(geo["height"]),
        "bands": 3,
        "band_mapping": {
            "B04": "Red",
            "B03": "Green",
            "B02": "Blue",
        },
        "crs": str(geo["crs"]),
        "transform": [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
            0.0,
            0.0,
            1.0,
        ],
        "bounds": [
            float(bounds.left),
            float(bounds.bottom),
            float(bounds.right),
            float(bounds.top),
        ],
        "resolution": [
            float(abs(geo["resolution"][0])),
            float(abs(geo["resolution"][1])),
        ],
        "processing": {
            "model": "Depth Anything V2",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_name": Path(CHECKPOINT).name,
            "model_size": int(INPUT_SIZE),
            "device": str(DEVICE),
            "learned_scale": float(scale),
            "learned_shift": float(shift),
        },
        "srtm": {
            "tiles": srtm_tiles,
            "source": SRTM_BASE,
            "aligned_to_input_grid": True,
        },
        "dsm": {
            "method": "SRTM + predicted height",
            "provisional": True,
            "negative_values_clipped": True,
            "pixels_clipped": int(pixels_clipped),
            "min": dsm_min,
            "max": dsm_max,
            "mean": dsm_mean,
            "median": dsm_median,
        },
        "threejs": {
            "texture": "texture.png",
            "heightmap": "heightmap.png",
            "heightmap_npy": "heightmap.npy",
            "width": int(geo["width"]),
            "height": int(geo["height"]),
            "elevation_min": dsm_min,
            "elevation_max": dsm_max,
        },
        "pipeline": {
            "started": started,
            "finished": finished,
            "input_file": str(input_path),
            "output_directory": str(run_dir),
        },
    }

    metadata_path = run_dir / "metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    return metadata


# ============================================================
# NORMAL IMAGE / GEOTIFF FALLBACK
# ============================================================

def process_regular_image(input_path: Path, run_dir: Path, started: str):
    """
    JPG/PNG/GeoTIFF fallback.

    For a georeferenced GeoTIFF we can still produce a terrain DSM using
    SRTM. For a plain JPG/PNG, there is no geographic position, so the
    model output itself becomes the provisional heightmap.
    """
    print("\n=========================================================================")
    print("IMAGE PIPELINE")
    print("=========================================================================")

    model, scale, shift = load_model()
    image_rgb, raster_metadata = load_image(str(input_path))

    agl = predict(
        model,
        image_rgb,
        scale,
        shift,
    )

    depth_tif = run_dir / "depth.tif"
    depth_npy = run_dir / "depth.npy"

    save_agl(
        agl,
        str(depth_tif),
        raster_metadata,
    )
    save_numpy(
        agl,
        str(depth_npy),
    )
    save_texture(
        image_rgb,
        run_dir / "texture.png",
    )

    georeferenced = (
        raster_metadata is not None
        and raster_metadata.get("crs") is not None
        and raster_metadata.get("transform") is not None
    )

    if georeferenced:
        with rasterio.open(depth_tif) as src:
            bounds = src.bounds
            crs = src.crs
            transform = src.transform
            resolution = src.res

        if str(crs).upper() == "EPSG:4326":
            lonlat_bounds = tuple(bounds)
        else:
            from rasterio.warp import transform_bounds

            lonlat_bounds = transform_bounds(
                crs,
                "EPSG:4326",
                *bounds,
                densify_pts=21,
            )

        srtm_raw = run_dir / "srtm_raw.tif"
        srtm_tiles = build_srtm_mosaic(
            lonlat_bounds,
            srtm_raw,
        )

        srtm_aligned = run_dir / "srtm_aligned.tif"
        align_dem_to_agl(
            srtm_raw,
            depth_tif,
            srtm_aligned,
        )

        dsm_tif = run_dir / "dsm.tif"
        generate_dsm(
            agl_path=str(depth_tif),
            dem_path=str(srtm_aligned),
            output_path=str(dsm_tif),
        )

        heightmap, pixels_clipped = clip_negative_dsm(dsm_tif)

        try:
            srtm_raw.unlink()
        except OSError:
            pass

        dsm_method = "SRTM + predicted height"
        absolute_elevation = True

    else:
        heightmap = agl.astype(np.float32)
        pixels_clipped = 0
        srtm_tiles = []
        dsm_method = "predicted height only"
        absolute_elevation = False
        crs = None
        transform = None
        bounds = None
        resolution = None

    np.save(
        run_dir / "heightmap.npy",
        heightmap.astype(np.float32),
    )

    save_heightmap_png(
        heightmap,
        run_dir / "heightmap.png",
    )

    valid = np.isfinite(heightmap)

    hmin = float(np.nanmin(heightmap))
    hmax = float(np.nanmax(heightmap))
    hmean = float(np.nanmean(heightmap))
    hmedian = float(np.nanmedian(heightmap))

    h, w = heightmap.shape

    metadata = {
        "input_type": "image",
        "input_kind": "geotiff_rgb" if georeferenced else "rgb_image",
        "georeferenced": bool(georeferenced),
        "width": int(w),
        "height": int(h),
        "bands": 3,
        "processing": {
            "model": "Depth Anything V2",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_name": Path(CHECKPOINT).name,
            "model_size": int(INPUT_SIZE),
            "device": str(DEVICE),
            "learned_scale": float(scale),
            "learned_shift": float(shift),
        },
        "dsm": {
            "method": dsm_method,
            "provisional": True,
            "absolute_elevation": bool(absolute_elevation),
            "negative_values_clipped": bool(georeferenced),
            "pixels_clipped": int(pixels_clipped),
            "min": hmin,
            "max": hmax,
            "mean": hmean,
            "median": hmedian,
        },
        "threejs": {
            "texture": "texture.png",
            "heightmap": "heightmap.png",
            "heightmap_npy": "heightmap.npy",
            "width": int(w),
            "height": int(h),
            "elevation_min": hmin,
            "elevation_max": hmax,
        },
        "pipeline": {
            "started": started,
            "finished": datetime.now().isoformat(),
            "input_file": str(input_path),
            "output_directory": str(run_dir),
        },
    }

    if georeferenced:
        metadata["crs"] = str(crs)
        metadata["transform"] = [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
            0.0,
            0.0,
            1.0,
        ]
        metadata["bounds"] = [
            float(bounds.left),
            float(bounds.bottom),
            float(bounds.right),
            float(bounds.top),
        ]
        metadata["resolution"] = [
            float(abs(resolution[0])),
            float(abs(resolution[1])),
        ]
        metadata["srtm"] = {
            "tiles": srtm_tiles,
            "source": SRTM_BASE,
            "aligned_to_input_grid": True,
        }

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    return metadata


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 73)
    print("SIH 26175 TERRAIN PIPELINE")
    print("=" * 73)

    input_path = get_input_path()
    run_dir = make_run_dir()
    started = datetime.now().isoformat()

    print(f"\nInput:\n  {input_path}")
    print(f"\nNew output run:\n  {run_dir}")

    try:
        if input_path.suffix.lower() == ".zip":
            metadata = process_sentinel_zip(
                input_path,
                run_dir,
                started,
            )
        else:
            metadata = process_regular_image(
                input_path,
                run_dir,
                started,
            )

    except Exception:
        # Keep the failed run for debugging, but mark it clearly.
        try:
            (run_dir / "FAILED.txt").write_text(
                "Pipeline failed. See terminal traceback.\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        raise

    print("\n=========================================================================")
    print("PIPELINE COMPLETE")
    print("=========================================================================")

    print(f"Input type : {metadata.get('input_type')}")
    print(f"Size       : {metadata.get('width')} x {metadata.get('height')}")
    print(f"Output     : {run_dir}")

    print("\nGenerated files:")

    for name in [
        "depth.npy",
        "depth.tif",
        "dsm.tif",
        "heightmap.npy",
        "heightmap.png",
        "metadata.json",
        "srtm_aligned.tif",
        "texture.png",
    ]:
        p = run_dir / name
        print(f"  {'✓' if p.exists() else '-'} {name}")

    # Critical safety check: never visualize some unrelated old run.
    if not (run_dir / "heightmap.npy").exists():
        raise RuntimeError(
            f"Current run did not generate heightmap.npy:\n{run_dir}"
        )

    print("\nLaunching 3D reconstruction for THIS run...")
    launch_3d_simulation_mesh = import_visualizer()
    launch_3d_simulation_mesh(str(run_dir))


if __name__ == "__main__":
    main()
