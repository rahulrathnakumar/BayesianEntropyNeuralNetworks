import matplotlib.pyplot as plt
import numpy as np
import porespy as ps
import inspect
inspect.signature(ps.metrics.two_point_correlation)

np.random.seed(10)
im = ps.generators.blobs(shape=[256, 256])

data = ps.metrics.two_point_correlation(im, bins = 75, voxel_size = 1)
print(data.probability, data.distance)

import torch
from torch_tpc import two_point_correlation

im = torch.from_numpy(im)

data_torch = two_point_correlation(im, bins = 75, voxel_size = 1, gpu_id=0)
print(data_torch.probability, data_torch.distance)


# compute mse between numpy and torch
mse = np.square(np.subtract(data.probability, data_torch.probability.detach().cpu().numpy())).mean()
print(mse)
