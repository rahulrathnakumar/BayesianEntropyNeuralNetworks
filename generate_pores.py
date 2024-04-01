import numpy as np
import porespy as ps  # PoreSpy library
from torch_tpc import two_point_correlation  # Replace 'your_module' with your actual module name

# Generate pores
# Step 1: Generate a synthetic image (or use your own data)
# Example: Generate a synthetic image using PoreSpy
shape = (200, 200)
# randomize porosity and blobiness to generate different images
porosity = np.random.uniform(0.2, 0.6)
blobiness = np.random.uniform(1, 3)

# Generate a synthetic dataset using Porespy - 1100 images
N = 1000
images = np.zeros((N, *shape))
for i in range(N):
    print("Generating image", i)
    images[i] = ps.generators.blobs(shape=shape, porosity=porosity, blobiness=blobiness)

# save the dataset as a numpy array 
np.save('data/generated/synthetic_images.npy', images)


