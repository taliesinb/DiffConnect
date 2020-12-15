from cache import cached, load_cached_results_as_pandas

import torch
from utils import *

import torch.distributions.uniform

def t(x):
    return torch.tensor(x, dtype=torch.float32)

def generate_grid(size, positions, sigma):
    ranges = [torch.linspace(-1.0, 1.0, d) for d in size]
    coords = torch.cartesian_prod(*ranges)
    dists = torch.cdist(coords, positions)
    y = torch.exp(-(dists / sigma) ** 2)
    return y.view(*size, positions.shape[0])

def uniform(*shape):
    return 2 * torch.rand(*shape) - 1

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PureSpatialXOXLinear(nn.Module):
    def __init__(self, i_shape, o_shape, dim, sigma_1, sigma_2):
        super().__init__()
        self.dim = dim
        self.i_pos = uniform(dim, 2)
        self.o_pos = uniform(dim, 2)
        self.i_grid = generate_grid(i_shape, self.i_pos, sigma_1)
        self.o_grid = generate_grid(o_shape, self.o_pos, sigma_2) # sz... * dim
        self.i_size = np.prod(i_shape)
        self.o_size = np.prod(o_shape)
        self.o_matrix = nn.Parameter(torch.randn(dim, dim) * (1 / np.sqrt(3 * self.i_size)))
        self.bias = nn.Parameter(torch.zeros(self.o_size))

    def forward(self, input):
        weight = self.yox(
            self.o_grid.flatten(end_dim=-2),
            self.i_grid.flatten(end_dim=-2)
        )
        return F.linear(input, weight, self.bias)

    def print_grids(self):
        print_image(self.i_grid.permute(2, 0, 1))
        print_image(self.o_grid.permute(2, 0, 1))

    # takes y = N * d and x = M * d and o_matrix = d x d and produces M * N
    def yox(self, y, x):
        res = torch.matmul(y, torch.matmul(self.o_matrix, torch.t(x)))
        return res

# spatial = PureSpatialXOXLinear([28, 28], [4, 4], 12, 0.5, 0.5)
# spatial.print_grids()

from train import train
from data import *

best_acc = 0

@cached
def final_accuracy_for_spatial_xox(sigma0, sigma1):
    acc = 0
    for _ in range(5):
        net = PureSpatialXOXLinear([28, 28], [2, 5], 20, sigma0, sigma1)
        res = train(net, mnist, max_batches=5000, log_dir=None, flatten=True)
        acc += res['accuracy']
    return acc / 5

sigmas = np.linspace(0.1, 0.9, 5)

for sigma0 in sigmas:
    for sigma1 in sigmas:
        acc = final_accuracy_for_spatial_xox(sigma0, sigma1)
        print(f'%{sigma0:.2f} %{sigma1:.2f} => %{acc:.3f}')