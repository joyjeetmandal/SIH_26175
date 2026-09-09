from gamus import GAMUSDataset


dataset = GAMUSDataset("../data/Gamus")

print("Dataset length:", len(dataset))

sample = dataset[0]

print("RGB shape:", sample["rgb"].shape)
print("AGL shape:", sample["agl"].shape)
print("Mask shape:", sample["mask"].shape)

print("First RGB file:", dataset.samples[0][0])
print("First AGL file:", dataset.samples[0][1])
