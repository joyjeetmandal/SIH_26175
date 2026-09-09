from pathlib import Path

import numpy as np
import rasterio


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AGL_FILE = PROJECT_ROOT / "outputs" / "depth" / "sentinel_depth_5004.npy"

SRTM_FILE = PROJECT_ROOT / "outputs" / "dem" / "srtm_aligned_to_sentinel.tif"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dsm"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = OUTPUT_DIR / "sentinel_provisional_dsm_5004.tif"


# ============================================================
# LOAD AGL
# ============================================================

print("=" * 70)
print("CREATING PROVISIONAL DSM")
print("=" * 70)

agl = np.load(AGL_FILE).astype(np.float32)


# ============================================================
# LOAD SRTM
# ============================================================

with rasterio.open(SRTM_FILE) as src:
    srtm = src.read(1).astype(np.float32)

    profile = src.profile.copy()


# ============================================================
# CHECK
# ============================================================

print("\nAGL shape: ", agl.shape)
print("SRTM shape:", srtm.shape)

if agl.shape != srtm.shape:
    raise ValueError("AGL and SRTM grids do not match.")


# ============================================================
# STATISTICS
# ============================================================

print("\nAGL:")
print("  Min:   ", agl.min())
print("  Max:   ", agl.max())
print("  Mean:  ", agl.mean())
print("  Median:", np.median(agl))


print("\nSRTM:")
print("  Min:   ", srtm.min())
print("  Max:   ", srtm.max())
print("  Mean:  ", srtm.mean())
print("  Median:", np.median(srtm))


# ============================================================
# CREATE DSM
# ============================================================

dsm = (srtm + agl).astype(np.float32)


# ============================================================
# DSM STATISTICS
# ============================================================

print("\nProvisional DSM:")
print("  Min:   ", dsm.min())
print("  Max:   ", dsm.max())
print("  Mean:  ", dsm.mean())
print("  Median:", np.median(dsm))


# ============================================================
# SAVE
# ============================================================

profile.update(
    {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "compress": "deflate",
    }
)


with rasterio.open(
    OUTPUT_FILE,
    "w",
    **profile,
) as dst:
    dst.write(
        dsm,
        1,
    )


print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("PROVISIONAL DSM COMPLETE")
print("=" * 70)
