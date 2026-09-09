import h5py

RGB_FILE = "Gamus/images/test/DC_03_26_RGB.h5"
AGL_FILE = "Gamus/heights/test/DC_03_26_AGL.h5"


def inspect_h5(path):
    print(f"\n===== {path} =====")

    with h5py.File(path, "r") as f:
        print("Top-level keys:", list(f.keys()))

        def show_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"\nDataset: {name}")
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                print(f"  Size:  {obj.size}")

        f.visititems(show_dataset)


inspect_h5(RGB_FILE)
inspect_h5(AGL_FILE)
