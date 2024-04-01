import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import AnisoDataset

import numpy as np

# Define the transformation to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

# Create the dataset and dataloader
dataset = AnisoDataset('data/aniso_data_huge', 'train', transforms=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Variables to store total sum and total count for images
total_sum = 0
total_count = 0

# Iterate over all images
for batch in dataloader:
    total_sum += batch.sum()
    total_count += np.prod(batch.size())

# Compute mean and standard deviation
mean = total_sum / total_count

sum_of_squared_error = 0
for batch in dataloader:
    sum_of_squared_error += ((batch - mean).pow(2)).sum()
std = torch.sqrt(sum_of_squared_error / total_count)

print(f'Mean: {mean.item()}')
print(f'Std: {std.item()}')