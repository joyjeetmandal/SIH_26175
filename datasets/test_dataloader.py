from gamus import GAMUSDataset
from torch.utils.data import DataLoader


dataset = GAMUSDataset("../data/Gamus")

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
)


print("Dataset length:", len(dataset))


for batch in loader:
    print("\nRGB batch shape:", batch["rgb"].shape)
    print("AGL batch shape:", batch["agl"].shape)
    print("Mask batch shape:", batch["mask"].shape)

    break
