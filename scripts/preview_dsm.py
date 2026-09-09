import numpy as np
import rasterio
import matplotlib.pyplot as plt

path = "outputs/dsm/sentinel_provisional_dsm_5004_clipped.tif"

with rasterio.open(path) as src:
    dsm = src.read(1)

print("Shape:", dsm.shape)
print("Min:", dsm.min())
print("Max:", dsm.max())
print("Mean:", dsm.mean())
print("Median:", np.median(dsm))

plt.figure(figsize=(10, 7))
plt.imshow(dsm)
plt.colorbar(label="Elevation (m)")
plt.title("Provisional DSM")
plt.tight_layout()

output = "outputs/dsm/provisional_dsm_preview.png"
plt.savefig(output, dpi=150)

print("Preview saved to:", output)

plt.show()
