from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SENTINEL_FILE = next((PROJECT_ROOT / "data" / "sentinel").glob("*B04*Raw*.tiff"))

SRTM_FILE = PROJECT_ROOT / "data" / "dem" / "N22E088.hgt"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dem"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = OUTPUT_DIR / "srtm_aligned_to_sentinel.tif"


# ============================================================
# INFORMATION
# ============================================================

print("=" * 70)
print("SRTM → SENTINEL GRID ALIGNMENT")
print("=" * 70)


# ============================================================
# OPEN SENTINEL REFERENCE
# ============================================================

print("\nOpening Sentinel reference...")

with rasterio.open(SENTINEL_FILE) as sentinel:
    sentinel_crs = sentinel.crs
    sentinel_transform = sentinel.transform
    sentinel_width = sentinel.width
    sentinel_height = sentinel.height
    sentinel_bounds = sentinel.bounds

    print(
        "Sentinel shape:",
        (
            sentinel_height,
            sentinel_width,
        ),
    )

    print(
        "Sentinel CRS:",
        sentinel_crs,
    )

    print(
        "Sentinel bounds:",
        sentinel_bounds,
    )

    print(
        "Sentinel resolution:",
        sentinel.res,
    )


# ============================================================
# OPEN SRTM
# ============================================================

print("\nOpening SRTM...")

with rasterio.open(SRTM_FILE) as srtm:
    print(
        "SRTM shape:",
        srtm.shape,
    )

    print(
        "SRTM CRS:",
        srtm.crs,
    )

    print(
        "SRTM bounds:",
        srtm.bounds,
    )

    print(
        "SRTM resolution:",
        srtm.res,
    )

    srtm_data = srtm.read(1)


# ============================================================
# CREATE OUTPUT ARRAY
# ============================================================

aligned_srtm = np.empty(
    (
        sentinel_height,
        sentinel_width,
    ),
    dtype=np.float32,
)


# ============================================================
# REPROJECT / RESAMPLE
# ============================================================

print("\nResampling SRTM onto Sentinel grid...")

reproject(
    source=srtm_data,
    destination=aligned_srtm,
    src_transform=srtm.transform,
    src_crs=srtm.crs,
    dst_transform=sentinel_transform,
    dst_crs=sentinel_crs,
    resampling=Resampling.bilinear,
)


# ============================================================
# STATISTICS
# ============================================================

print("\nAligned SRTM statistics:")

print(
    "Min:",
    np.nanmin(aligned_srtm),
)

print(
    "Max:",
    np.nanmax(aligned_srtm),
)

print(
    "Mean:",
    np.nanmean(aligned_srtm),
)

print(
    "Median:",
    np.nanmedian(aligned_srtm),
)


# ============================================================
# SAVE
# ============================================================

profile = {
    "driver": "GTiff",
    "height": sentinel_height,
    "width": sentinel_width,
    "count": 1,
    "dtype": "float32",
    "crs": sentinel_crs,
    "transform": sentinel_transform,
    "compress": "deflate",
}


with rasterio.open(
    OUTPUT_FILE,
    "w",
    **profile,
) as dst:
    dst.write(
        aligned_srtm,
        1,
    )


# ============================================================
# VERIFY
# ============================================================

print("\nSaved:")

print(OUTPUT_FILE)


with rasterio.open(OUTPUT_FILE) as src:
    print("\nVerification:")

    print(
        "Shape:",
        src.shape,
    )

    print(
        "CRS:",
        src.crs,
    )

    print(
        "Resolution:",
        src.res,
    )

    print(
        "Bounds:",
        src.bounds,
    )

    print(
        "Transform:",
        src.transform,
    )


print("\n" + "=" * 70)
print("ALIGNMENT COMPLETE")
print("=" * 70)
