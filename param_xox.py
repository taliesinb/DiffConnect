from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.distributions.uniform

import numpy as np
import torch.nn as nn
import torch.distributions.uniform

import utils

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

class GaussianExpression(GeneExpressionGenerator):

    def __init__(self, shape, sigma=0.3):
        self.sigma = sigma
        self.shape = shape

    def make_filters(self, positions, sigma):
        return utils.gaussian_filters(self.shape, positions, sigma)

class RandomGaussianExpression(GaussianExpression):

    def generate(self):
        self.constant_positions = uniform(self.num_genes, len(self.shape)) 
        return self.make_filters(self.constant_positions, 1/self.sigma)


class LearnedPositionGaussianExpression(GaussianExpression):

    def generate(self):
        self.learned_positions = nn.Parameter(uniform(self.num_genes, len(self.shape))) # positions of gaussians
        self.register_parameter('gaussian_positions', self.learned_positions)
        return self.forward()

    def forward(self):
        return self.make_filters(self.learned_positions, 1/self.sigma)

class LearnedGaussianExpression(GaussianExpression):

    def generate(self):
        self.learned_positions = nn.Parameter(uniform(self.num_genes, len(self.shape)))
        self.learned_log_sigmas = nn.Parameter(-np.log(self.sigma) * torch.ones(self.num_genes))
        self.register_parameter('gaussian_positions', self.learned_positions)
        self.register_parameter('gaussian_log_sigmas', self.learned_log_sigmas)
        return self.forward()

    def forward(self):
        return self.make_filters(self.learned_positions, torch.exp(self.learned_log_sigmas))

def to_size(size):
    return size if isinstance(size, int) else utils.product(size)

class LearnedExpression(GeneExpressionGenerator):

    def __init__(self, size):
        self.size = to_size(size)

    def generate(self):
        gene_matrix = nn.Parameter(torch.randn(self.size, self.num_genes))
        self.register_parameter('gene_matrix', gene_matrix)
        return gene_matrix

class RandomExpression(GeneExpressionGenerator):

    def __init__(self, size):
        self.size = to_size(size)

    def generate(self):
        gene_matrix = torch.randn(self.size, self.num_genes)
        return gene_matrix

class XOXLinear(nn.Module):
    def __init__(self, i_expr, o_expr, num_genes, learned_bias=False):
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
        if learned_bias:
            self.bias = nn.Parameter(torch.zeros(o_size // num_genes))
        else:
            self.bias = torch.randn(o_size // num_genes) * 0.5

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