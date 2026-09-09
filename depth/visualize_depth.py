from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEPTH_FILE = PROJECT_ROOT / "outputs" / "depth" / "sentinel_depth_5004.npy"

OUTPUT_FILE = PROJECT_ROOT / "outputs" / "depth" / "sentinel_depth_5004.png"


depth = np.load(DEPTH_FILE)


print("Shape:", depth.shape)
print("Min:", depth.min())
print("Max:", depth.max())
print("Mean:", depth.mean())
print("Median:", np.median(depth))


plt.figure(figsize=(10, 7))

plt.imshow(depth)

plt.colorbar(label="Predicted height/depth")

plt.title("Depth Anything V2 — Sentinel-2 Prediction")

plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print("\nVisualization saved to:")

print(OUTPUT_FILE)
