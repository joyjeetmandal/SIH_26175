import h5py
import matplotlib.pyplot as plt


RGB_FILE = "Gamus/images/test/DC_03_26_RGB.h5"
AGL_FILE = "Gamus/heights/test/DC_03_26_AGL.h5"


with h5py.File(RGB_FILE, "r") as f:
    rgb = f["image"][:]

with h5py.File(AGL_FILE, "r") as f:
    agl = f["image"][:]


print("RGB:", rgb.shape, rgb.dtype)
print("AGL:", agl.shape, agl.dtype)
print("AGL min:", agl.min())
print("AGL max:", agl.max())
print("AGL mean:", agl.mean())


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title("GAMUS RGB")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(agl)
plt.title("GAMUS AGL")
plt.colorbar(label="Height (m)")
plt.axis("off")

plt.tight_layout()
plt.savefig("gamus_sample.png", dpi=150)

print("Saved: gamus_sample.png")
