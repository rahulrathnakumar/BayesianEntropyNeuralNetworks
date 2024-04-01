# Example - Convolutional DBM using PyTorch-ProbGraph
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_probgraph import BernoulliLayer, DiracDeltaLayer, CategoricalLayer
from pytorch_probgraph import GaussianLayer
from pytorch_probgraph import InteractionLinear, InteractionModule, InteractionPoolMapIn2D, InteractionPoolMapOut2D
from pytorch_probgraph import InteractionSequential
from pytorch_probgraph import RestrictedBoltzmannMachinePCD
from pytorch_probgraph import DeepBeliefNetwork
from itertools import chain
from tqdm import tqdm


# class Model_DBN_Conv_ProbMaxPool(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         layer0 = BernoulliLayer(torch.zeros([1, 1, 200, 200], requires_grad=True))
#         layer1 = CategoricalLayer(torch.zeros([1, 24, 97, 97, 5], requires_grad=True))
#         layer2 = CategoricalLayer(torch.zeros([1, 40, 44, 44, 5], requires_grad=True))

#         interaction0 = InteractionSequential(InteractionModule(torch.nn.Conv2d(1,24,6)), InteractionPoolMapIn2D(2, 2))
#         interaction1 = InteractionSequential(InteractionPoolMapOut2D(2,2), InteractionModule(torch.nn.Conv2d(24,40,6)), InteractionPoolMapIn2D(2, 2))

#         rbm1 = RestrictedBoltzmannMachinePCD(layer0, layer1, interaction0, fantasy_particles=10)
#         rbm2 = RestrictedBoltzmannMachinePCD(layer1, layer2, interaction1, fantasy_particles=10)
#         opt = torch.optim.Adam(chain(rbm1.parameters(), rbm2.parameters()), lr=1e-3)
#         self.model = DeepBeliefNetwork([rbm1, rbm2], opt)
#         #self.model = self.model.to(device)
#         #print(interaction.weight.shape)

#     def train(self, data, epochs=1, device=None):
#         self.model.train(data, epochs=epochs, device=device)

#     def loglikelihood(self, data):
#         data = data.reshape(-1, 1, 1, 28, 28)
#         return -self.model.free_energy_estimate(data)

#     def generate(self, N=1):
#         return self.model.sample(N=N, gibbs_steps=100).cpu()

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



if __name__ == '__main__':
    # create model instance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model_DBN_Conv_ProbMaxPool()
    model = model.to(device)
    # create a random image 
    x = torch.rand(1, 16, 1, 28, 28).to(device)
    # train the model for 1 epoch
    model.train(x, epochs=2, device = device)
    print("Training Complete")

    # generate a sample
    sample = model.generate(1)
    print('generated sample shape: ', sample.shape)