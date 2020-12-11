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

spatial = PureSpatialXOXLinear([28, 28], [4, 4], 12, 0.5, 0.5)
spatial.print_grids()

from train import train
from data import *

best_acc = 0
for i in range(5):
    net = nn.Sequential(
        PureSpatialXOXLinear([28, 28], [4, 4], 10, 0.3, 0.6), nn.Tanh(),
        #PureSpatialXOXLinear([10, 10], [4, 4], 5, 0.6, 0.7), nn.Tanh(),
        nn.Linear(16, 10)
    )
    if i == 0: net[0].print_grids()
    res = train(net, mnist, max_batches=3000, log_dir=None, flatten=True)

