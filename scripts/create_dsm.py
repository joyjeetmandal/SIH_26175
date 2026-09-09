import numpy as np
import rasterio
from pathlib import Path

PROJECT_ROOT = Path("/mnt/Games_codes/python_development/SIH/SIH_26175")

srtm_path = PROJECT_ROOT / "outputs/dem/srtm_aligned_to_sentinel.tif"
agl_path = PROJECT_ROOT / "outputs/depth/sentinel_depth_5004.tif"
output_path = PROJECT_ROOT / "outputs/dsm/sentinel_provisional_dsm_5004_clipped.tif"

with rasterio.open(srtm_path) as src:
    srtm = src.read(1)
    profile = src.profile.copy()

with rasterio.open(agl_path) as src:
    agl = src.read(1)

# Provisional DSM
dsm = srtm + agl

# Arbitrary floor at 0 m
dsm = np.maximum(dsm, 0.0).astype(np.float32)

profile.update(dtype="float32", count=1, compress="deflate")

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(dsm, 1)

print("Saved:", output_path)
print("Shape:", dsm.shape)
print("Min:", dsm.min())
print("Max:", dsm.max())
print("Mean:", dsm.mean())
print("Median:", np.median(dsm))
print("Pixels clipped:", np.sum((srtm + agl) < 0))
