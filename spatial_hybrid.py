from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.distributions.uniform

import numpy as np
import torch.nn as nn
import torch.distributions.uniform

def uniform(*shape):
    return 2 * torch.rand(*shape) - 1

class GeneExpressionGenerator:

    def attach(self, parent, name : str, num_genes : int):
        self.parent = parent
        self.name = name
        self.num_genes = num_genes

    def register_parameter(self, param_name : str, array):
        self.parent.register_parameter(self.name + ":" + param_name, array)


class GaussianExpression(GeneExpressionGenerator):

    def __init__(self, shape, sigma=0.3):
        self.sigma = sigma
        self.shape = shape

    def generate(self):
        positions = uniform(self.num_genes, len(self.shape)) # positions of gaussians
        ranges = [torch.linspace(-1.0, 1.0, d) for d in self.shape]
        coords = torch.cartesian_prod(*ranges)
        dists = torch.cdist(coords, positions)
        y = torch.exp(-(dists / self.sigma) ** 2)
        return y.view(*self.shape, positions.shape[0])

class LearnedExpression(GeneExpressionGenerator):

    def __init__(self, size):
        self.size = size

    def generate(self):
        gene_matrix = nn.Parameter(torch.randn(self.size, self.num_genes))
        self.register_parameter('gene_matrix', gene_matrix)
        return gene_matrix

class RandomExpression(GeneExpressionGenerator):

    def __init__(self, size):
        self.size = size

    def generate(self):
        gene_matrix = torch.randn(self.size, self.num_genes)
        return gene_matrix


class ParameterizedXOXLinear(nn.Module):
    def __init__(self, i_expr, o_expr, num_genes):
        super().__init__()
        i_expr.attach(self, 'input', num_genes)
        o_expr.attach(self, 'input', num_genes)
        self.i_array = i_expr.generate()
        self.o_array = o_expr.generate()
        i_size = self.i_array.numel()
        o_size = self.o_array.numel()
        self.o_matrix = nn.Parameter(torch.randn(num_genes, num_genes) * (1 / np.sqrt(3 * i_size)))
        self.bias = nn.Parameter(torch.zeros(o_size // num_genes))

    def forward(self, vec):
        weight = self.yox(
            self.o_array.flatten(end_dim=-2),
            self.i_array.flatten(end_dim=-2)
        )
        return F.linear(vec, weight, self.bias)

    # takes y = N * d and x = M * d and o_matrix = d x d and produces M * N
    def yox(self, y, x):
        res = torch.matmul(y, torch.matmul(self.o_matrix, torch.t(x)))
        return res

from train import train
from data import *

print("\nlearned to random")
net = ParameterizedXOXLinear(LearnedExpression(28*28), RandomExpression(10), 5)
res = train(net, mnist, max_batches=10000, log_dir=None, flatten=True)

print("\nrandom to learned")
net = ParameterizedXOXLinear(RandomExpression(28*28), LearnedExpression(10), 5)
res = train(net, mnist, max_batches=10000, log_dir=None, flatten=True)

print("\nrandom to random")
net = ParameterizedXOXLinear(RandomExpression(28*28), RandomExpression(10), 5)
res = train(net, mnist, max_batches=10000, log_dir=None, flatten=True)

print("\ngaussian to learned")
net = ParameterizedXOXLinear(GaussianExpression([28, 28]), LearnedExpression(10), 5)
res = train(net, mnist, max_batches=10000, log_dir=None, flatten=True)

print("\ngaussian to random")
net = ParameterizedXOXLinear(GaussianExpression([28, 28]), RandomExpression(10), 5)
res = train(net, mnist, max_batches=10000, log_dir=None, flatten=True)

# @cached
# def final_accuracy_for_parameterized_xox(sigma):
#     acc = 0
#     for _ in range(5):
#         net = PureSpatialXOXLinear([28, 28], [2, 5], 20, sigma0, sigma1)
#         res = train(net, mnist, max_batches=5000, log_dir=None, flatten=True)
#         acc += res['accuracy']
#     return acc / 5

# for sigma in np.arange(0.1, 0.9, 0.05):
#     print(final_accuracy_for_parameterized_xox(sigma))