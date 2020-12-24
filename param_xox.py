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

    def forward(self):
        pass

def gaussian_expression(positions, shape, sigma):
    ranges = [torch.linspace(-1.0, 1.0, d) for d in shape]
    coords = torch.cartesian_prod(*ranges)
    dists = torch.cdist(coords, positions)
    y = torch.exp(-(dists / sigma) ** 2)
    return y.view(*shape, positions.shape[0])

class GaussianExpression(GeneExpressionGenerator):

    def __init__(self, shape, sigma=0.3):
        self.sigma = sigma
        self.shape = shape

    def generate(self):
        positions = uniform(self.num_genes, len(self.shape)) # positions of gaussians
        return gaussian_expression(positions, self.shape, self.sigma)


class LearnedGaussianExpression(GaussianExpression):

    def generate(self):
        self.positions = nn.Parameter(uniform(self.num_genes, len(self.shape))) # positions of gaussians
        self.register_parameter('gaussian_positions', self.positions)
        return gaussian_expression(self.positions, self.shape, self.sigma)

    def forward(self):
        return gaussian_expression(self.positions, self.shape, self.sigma)

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
        self.i_forward = i_expr.forward
        self.o_forward = o_expr.forward
        self.o_matrix = nn.Parameter(torch.randn(num_genes, num_genes) * (1 / np.sqrt(3 * i_size)))
        self.bias = nn.Parameter(torch.zeros(o_size // num_genes))

    def forward(self, vec):
        new_i_array = self.i_forward()
        new_o_array = self.o_forward()
        if new_i_array is not None: self.i_array = new_i_array
        if new_o_array is not None: self.o_array = new_o_array
        weight = self.yox(
            self.o_array.flatten(end_dim=-2),
            self.i_array.flatten(end_dim=-2)
        )
        return F.linear(vec, weight, self.bias)

    # takes y = N * d and x = M * d and o_matrix = d x d and produces M * N
    def yox(self, y, x):
        res = torch.matmul(y, torch.matmul(self.o_matrix, torch.t(x)))
        return res
