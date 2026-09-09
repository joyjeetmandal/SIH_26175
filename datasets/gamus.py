from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class GAMUSDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

        self.rgb_dir = self.root_dir / "images"
        self.agl_dir = self.root_dir / "heights"

        self.samples = []

        for rgb_file in sorted(self.rgb_dir.rglob("*_RGB.h5")):
            name = rgb_file.name.replace("_RGB.h5", "_AGL.h5")

            agl_file = self.agl_dir / rgb_file.parent.relative_to(self.rgb_dir) / name

            if agl_file.exists():
                self.samples.append((rgb_file, agl_file))

        if not self.samples:
            raise RuntimeError("No RGB/AGL pairs found.")

        print(f"Found {len(self.samples)} RGB/AGL pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        rgb_file, agl_file = self.samples[index]

        with h5py.File(rgb_file, "r") as f:
            rgb = f["image"][:]

        with h5py.File(agl_file, "r") as f:
            agl = f["image"][:]

        # H x W x C → C x H x W
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)

        # H x W → 1 x H x W
        agl = torch.from_numpy(agl).unsqueeze(0)

        # Ignore negative/invalid AGL values
        mask = agl >= 0

        return {
            "rgb": rgb,
            "agl": agl,
            "mask": mask,
        }
