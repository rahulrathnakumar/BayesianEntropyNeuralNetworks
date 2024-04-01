import numpy as np
import torch
import unittest

def indices_numpy(dimensions):
    return np.indices(dimensions)

def indices_torch(dimensions):
    grids = torch.meshgrid([torch.arange(d) for d in dimensions])
    return grids

class TestIndices(unittest.TestCase):
    def test_numpy_vs_torch(self):
        # Define dimensions for the grid
        dimensions = (3, 4, 5)
        
        # Calculate grid coordinates using NumPy and PyTorch implementations
        numpy_indices = indices_numpy(dimensions)
        torch_indices = indices_torch(dimensions)

        
        # Ensure that the results are equal
        for i in range(len(dimensions)):
            np.testing.assert_array_equal(numpy_indices[i], torch_indices[i].numpy())

if __name__ == '__main__':
    unittest.main()
