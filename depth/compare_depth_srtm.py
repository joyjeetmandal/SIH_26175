from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = PROJECT_ROOT / "outputs" / "depth" / "sentinel_depth_5004.npy"

SRTM_FILE = PROJECT_ROOT / "outputs" / "dem" / "srtm_aligned_to_sentinel.tif"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dem"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MODEL ↔ SRTM COMPARISON")
print("=" * 70)

model = np.load(MODEL_FILE).astype(np.float32)


with rasterio.open(SRTM_FILE) as src:
    srtm = src.read(1).astype(np.float32)

    profile = src.profile.copy()


# ============================================================
# CHECK SHAPES
# ============================================================

print("\nModel shape:", model.shape)
print("SRTM shape: ", srtm.shape)


if model.shape != srtm.shape:
    raise ValueError("Model and SRTM shapes do not match.")


# ============================================================
# VALID MASK
# ============================================================

valid = np.isfinite(model) & np.isfinite(srtm)


model_valid = model[valid]
srtm_valid = srtm[valid]


# ============================================================
# BASIC STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("MODEL STATISTICS")
print("=" * 70)

print("Min:    ", model_valid.min())

print("Max:    ", model_valid.max())

print("Mean:   ", model_valid.mean())

print("Median: ", np.median(model_valid))


print("\n" + "=" * 70)
print("SRTM STATISTICS")
print("=" * 70)

print("Min:    ", srtm_valid.min())

print("Max:    ", srtm_valid.max())

print("Mean:   ", srtm_valid.mean())

print("Median: ", np.median(srtm_valid))


# ============================================================
# PEARSON CORRELATION
# ============================================================

model_centered = model_valid - model_valid.mean()

srtm_centered = srtm_valid - srtm_valid.mean()


numerator = np.sum(model_centered * srtm_centered)


denominator = np.sqrt(np.sum(model_centered**2) * np.sum(srtm_centered**2))


if denominator > 0:
    correlation = numerator / denominator

else:
    correlation = 0.0


print("\n" + "=" * 70)
print("CORRELATION")
print("=" * 70)

print(f"Pearson correlation: {correlation:.6f}")


# ============================================================
# LINEAR REGRESSION
# ============================================================

# Model prediction ≈ a × SRTM + b

a, b = np.polyfit(
    srtm_valid,
    model_valid,
    1,
)


print("\n" + "=" * 70)
print("LINEAR RELATIONSHIP")
print("=" * 70)

print(f"Model ≈ {a:.6f} × SRTM + {b:.6f}")


# ============================================================
# RMSE BETWEEN MODEL AND SRTM
# ============================================================

difference = model_valid - srtm_valid


rmse = np.sqrt(np.mean(difference**2))


mae = np.mean(np.abs(difference))


print("\n" + "=" * 70)
print("DIRECT DIFFERENCE")
print("=" * 70)

print(f"MAE:  {mae:.4f} m")

print(f"RMSE: {rmse:.4f} m")


# ============================================================
# SAVE DIFFERENCE MAP
# ============================================================

difference_map = (model - srtm).astype(np.float32)


difference_file = OUTPUT_DIR / "model_minus_srtm.tif"


profile.update(
    {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "compress": "deflate",
    }
)


with rasterio.open(
    difference_file,
    "w",
    **profile,
) as dst:
    dst.write(
        difference_map,
        1,
    )


print("\nDifference map saved:")

print(difference_file)


# ============================================================
# VISUALIZE SRTM
# ============================================================

plt.figure(figsize=(10, 7))

plt.imshow(srtm)

plt.colorbar(label="SRTM elevation (m)")

plt.title("SRTM Elevation")

plt.axis("off")

plt.tight_layout()

srtm_png = OUTPUT_DIR / "srtm_aligned.png"

plt.savefig(
    srtm_png,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# VISUALIZE DIFFERENCE
# ============================================================

plt.figure(figsize=(10, 7))

plt.imshow(difference_map)

plt.colorbar(label="Model − SRTM (m)")

plt.title("Model Prediction − SRTM")

plt.axis("off")

plt.tight_layout()

difference_png = OUTPUT_DIR / "model_minus_srtm.png"

plt.savefig(
    difference_png,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)

print("SRTM visualization:")

print(srtm_png)

print("\nDifference visualization:")

print(difference_png)

print("=" * 70)
