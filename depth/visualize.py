import cv2
import numpy as np

INPUT = "outputs/depth/demo19.npy"
OUTPUT = "outputs/depth/demo19_visual.png"

depth = np.load(INPUT)

# Normalize only for visualization
depth_min = depth.min()
depth_max = depth.max()

depth_norm = (depth - depth_min) / (depth_max - depth_min)
depth_norm = (depth_norm * 255).astype(np.uint8)

cv2.imwrite(OUTPUT, depth_norm)

print("Saved:", OUTPUT)
