import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import numpy as np
import pdb

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F

import matplotlib.pyplot as plt
from itertools import chain

from pytorch_probgraph import BernoulliLayer, DiracDeltaLayer, CategoricalLayer
from pytorch_probgraph import GaussianLayer
from pytorch_probgraph import InteractionLinear, InteractionModule, InteractionPoolMapIn2D, InteractionPoolMapOut2D
from pytorch_probgraph import InteractionSequential
from pytorch_probgraph import RestrictedBoltzmannMachinePCD
from pytorch_probgraph import DeepBeliefNetwork


class Model_DBN_Conv_ProbMaxPool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer0 = BernoulliLayer(torch.zeros([1, 1, 28, 28], requires_grad=True))
        layer1 = CategoricalLayer(torch.zeros([1, 40, 8, 8, 5], requires_grad=True))
        layer2 = CategoricalLayer(torch.zeros([1, 40, 1, 1, 5], requires_grad=True))

        interaction0 = InteractionSequential(InteractionModule(torch.nn.Conv2d(1,40,12)), InteractionPoolMapIn2D(2, 2))
        interaction1 = InteractionSequential(InteractionPoolMapOut2D(2,2), InteractionModule(torch.nn.Conv2d(40,40,6)), InteractionPoolMapIn2D(2, 2))

        rbm1 = RestrictedBoltzmannMachinePCD(layer0, layer1, interaction0, fantasy_particles=10)
        rbm2 = RestrictedBoltzmannMachinePCD(layer1, layer2, interaction1, fantasy_particles=10)
        opt = torch.optim.Adam(chain(rbm1.parameters(), rbm2.parameters()), lr=1e-3)
        self.model = DeepBeliefNetwork([rbm1, rbm2], opt)
        #self.model = self.model.to(device)
        #print(interaction.weight.shape)

    def train(self, data, epochs=1, device=None):
        self.model.train(data, epochs=epochs, device=device)

    def loglikelihood(self, data):
        data = data.reshape(-1, 1, 1, 28, 28)
        return -self.model.free_energy_estimate(data)

    def generate(self, N=1):
        return self.model.sample(N=N, gibbs_steps=100).cpu()

# Model Hyperparameters

dataset_path = 'ConstrainedUQ/data/alloy'
# dataset_path = 'data/alloy'

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

batch_size = 1024

xys = 28
x_dim  = xys*xys
hidden_dim = 784
latent_dim = 28

# Alloy dataset - MATLAB .mat file to numpy array

# load the .mat file 
import scipy.io as sio
mat_contents = sio.loadmat(dataset_path + '/alloy2.mat')

# convert to numpy array
xtr = mat_contents['xtr']

# Create dataset from xtr by creating N WxH patches by randomly sampling from xtr 
N = 1000
W = 28
H = 28

# sample 28x28 patches from xtr 
patches = np.zeros((N,W,H))
xys = np.zeros((N,2))
for i in range(N):
    x = np.random.randint(0, xtr.shape[1]-W)
    y = np.random.randint(0, xtr.shape[0]-H)
    # store the (x,y) coord of the patch to plot later
    xys[i,:] = np.array([x,y])
    patches[i,:,:] = xtr[y:y+W,x:x+H]

# plot some patches
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.imshow(patches[i,:,:], cmap='gray')
#     plt.axis('off')
# plt.show()

# convert to torch tensor
patches = torch.from_numpy(patches).float()
patches = patches.view(-1, 1, W, H)

# Create train loader
from torch.utils.data import Dataset, DataLoader

class TensorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Assuming self.data[index] is a single data point
        sample = self.data[index]
        return torch.tensor(sample)


train_dataset = TensorDataset(patches)
# Create train loader - it should not be list of tensors
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Create test loader by sampling 100 patches from xtr
N = 100
W = 28
H = 28

# sample 28x28 patches from xtr
patches = np.zeros((N,W,H))
xys = np.zeros((N,2))

for i in range(N):
    x = np.random.randint(0, xtr.shape[1]-W)
    y = np.random.randint(0, xtr.shape[0]-H)
    # store the (x,y) coord of the patch to plot later
    xys[i,:] = np.array([x,y])
    patches[i,:,:] = xtr[y:y+W,x:x+H]


# convert to torch tensor
patches = torch.from_numpy(patches).float()
patches = patches.view(-1, 1, W, H)

# Create test loader
test_dataset = TensorDataset(patches)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = Model_DBN_Conv_ProbMaxPool()
model = model.to(DEVICE)

# Train the model
model.train(train_loader, epochs=100, device=DEVICE)

# generate sample
sample = model.generate(N=1)
print("Completed generation of sample")

