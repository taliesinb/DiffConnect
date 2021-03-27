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
        if getattr(self, 'parent', None):
            return self.expressions
        self.name = name
        self.num_genes = num_genes
        self.parent = parent
        self.expressions = self.generate()
        return self.expressions

    def register_parameter(self, param_name : str, array):
        self.parent.register_parameter(self.name + ":" + param_name, array)

    def generate(self):
        raise NotImplementedError("GeneExpressionGenerator is a virtual base class")

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
        return self.make_filters(torch.tanh(self.learned_positions), 1/self.sigma)

class LearnedGaussianExpression(LearnedPositionGaussianExpression):

    def generate(self):
        self.learned_positions = nn.Parameter(torch.atanh(uniform(self.num_genes, len(self.shape)))) # positions of gaussians
        self.register_parameter('gaussian_positions', self.learned_positions)
        self.learned_sigmas = nn.Parameter(-np.log(self.sigma) * torch.ones(self.num_genes))
        self.register_parameter('gaussian_sigmas', self.learned_sigmas)
        return super().generate()

    def forward(self):
        return self.make_filters(torch.tanh(self.learned_positions), torch.exp(self.learned_sigmas))

    def dump_gaussian_parameters(self):
        pos = torch.tanh(self.learned_positions)
        sigmas = torch.exp(self.learned_sigmas)
        return torch_to_numpy(torch.hstack((pos, sigmas.unsqueeze(1))))

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

'''
ProteinInteractions can be attached to an XOXLinear when it is *created*,
in which case the corresponding o_matrix will be owned by that layer, or
they can be attached by a parent XOXSequential, which will own the o_matrix.
'''

'''
class LinearProteinInteractions
'''


class ProteinInteraction:

    def __init__(self, num_genes, include_bias=False):
        self.num_genes = num_genes
        self.include_bias = include_bias
        if include_bias: num_genes += 1
        self.biaser = utils.biaser(include_bias)
        self.o_matrix = nn.Parameter(torch.randn(num_genes, num_genes))

    def register_parameters(self, name, parent):
        parent.register_parameter(name + ':o_matrix', self.o_matrix)

    # takes y = N * d and x = M * d and o_matrix = d x d and produces M * N
    def calculate_weights(self, y, x):
        return torch.matmul(self.biaser(y), torch.matmul(self.o_matrix, torch.t(self.biaser(x))))


class RelabeledProteinInteraction(ProteinInteraction):

    def __init__(self, num_genes, num_labels, nonlinearity='tanh', include_bias=False):
        super().__init__(num_labels, include_bias)
        self.num_genes = num_genes
        self.num_labels = num_labels
        self.nonlinearity = utils.Nonlinearity(nonlinearity)
        if include_bias: num_genes += 1
        self.relabeling_matrix = nn.Parameter(torch.randn(num_genes, num_labels))

    def register_parameters(self, name, parent):
        super().register_parameters(name, parent)
        parent.register_parameter(name + ':relabeling_matrix', self.relabeling_matrix)

    # takes y = N * d and x = M * d and o_matrix = d x d and produces M * N
    def relabel(self, x):
        return self.nonlinearity(torch.matmul(self.biaser(x), self.relabeling_matrix))

    def calculate_weights(self, y, x):
        return super().calculate_weights(self.relabel(y), self.relabel(x))

def MaybeRelabeledProteinInteraction(num_genes, num_labels, nonlinearity='Tanh', include_bias=False):
    if num_labels is None or num_labels == 0:
        return ProteinInteraction(num_genes, include_bias=include_bias)
    else:
        return RelabeledProteinInteraction(num_genes, num_labels, nonlinearity, include_bias=include_bias)

class XOXSequential(nn.Sequential):
    def __init__(self, *args, interaction=None):
        super().__init__(*args)
        self.interaction = interaction
        if interaction:
            for module in self.modules():
                if hasattr(module, 'attach_interaction'):
                    module.attach_interaction(interaction)
            self.interaction.register_parameters('interaction', self)

def std(array):
    return max(array.std().item(), 1e-9)

# this is calculating the scaling factor to make the gnerated weights
# have a desired distribution empirically, so we simply *know* its right.
# there will be some noise from run to run, but its just an initialization
def calculate_reference_scaling(weights, init_fn=torch.nn.init.kaiming_normal_):
    ref_weights = torch.empty(weights.size())
    init_fn(ref_weights)
    return std(ref_weights) / std(weights)

class XOXLinear(nn.Module):
    def __init__(self, i_expr, o_expr, interaction=None, learned_bias=False, non_negative=False):
        super().__init__()
        self.i_expr = i_expr
        self.o_expr = o_expr
        self.learned_bias = learned_bias
        self.non_negative = non_negative
        if isinstance(interaction, int):
            interaction = ProteinInteraction(interaction)
        self.interaction = interaction
        if interaction:
            self.attach_interaction(interaction)
            interaction.register_parameters('interaction', self)

    def attach_interaction(self, interaction):
        self.interaction = interaction
        num_genes = interaction.num_genes
        self.i_array = self.i_expr.attach(self, 'input', num_genes)
        self.o_array = self.o_expr.attach(self, 'output', num_genes)
        # we divide by num_genes rather than just take the first dimension
        # since the {i,o}_arrays, since can be higher-rank thank two
        i_size = self.i_array.numel() // num_genes
        o_size = self.o_array.numel() // num_genes
        self.i_forward = self.i_expr.forward
        self.o_forward = self.o_expr.forward
        if self.learned_bias:
            self.bias = nn.Parameter(torch.zeros(o_size))
        else:
            self.bias = torch.randn(o_size) * 0.5
        self.scaling = 1.0 # so that self.calculate_weight() will work for the reference calculation
        self.scaling = calculate_reference_scaling(self.calculate_weights())

    def calculate_weights(self):
        if not self.interaction: raise RuntimeError("No protein interaction")
        new_i_array = self.i_forward()
        new_o_array = self.o_forward()
        if new_i_array is not None: self.i_array = new_i_array
        if new_o_array is not None: self.o_array = new_o_array
        weights = self.scaling * self.interaction.calculate_weights(
            self.o_array.flatten(end_dim=-2),
            self.i_array.flatten(end_dim=-2)
        )
        if self.non_negative: weights = F.relu(weights)
        return weights

    def forward(self, vec):
        return F.linear(vec, self.calculate_weights(), self.bias)

MaybeLearnedGaussianExpression = {True: LearnedGaussianExpression, False: RandomGaussianExpression}
MaybeLearnedExpression = {True: LearnedExpression, False: RandomExpression}
MaybeLearnedMaybeGaussianExpression = {True: MaybeLearnedGaussianExpression, False: MaybeLearnedExpression}

def maybe_shared(is_shared:bool, fn):
    if is_shared:
        e = e = fn()
        return e, e
    else:
        return fn(), fn()

'''
from utils import *
from train import train
from data import mnist

ishape = (28, 28)
oshape = 10

expr = RandomGaussianExpression([2,3])

print('\n\nautomatically global interaction:')
model = XOXSequential(
    XOXLinear(LearnedExpression(ishape), expr),
    XOXLinear(expr, LearnedExpression(oshape)),
    interaction=ProteinInteraction(5)
)

print_model_parameters(model)
train(model, mnist)

print('\n\nmanually global interaction')
interaction = ProteinInteraction(5)
model = nn.Sequential(
    XOXLinear(LearnedExpression(ishape), expr, interaction),
    XOXLinear(expr, LearnedExpression(oshape), interaction)
)

print_model_parameters(model)
train(model, mnist)

print('\n\nlocal interactions')
model = nn.Sequential(
    XOXLinear(LearnedExpression(ishape), expr, ProteinInteraction(5)),
    XOXLinear(expr, LearnedExpression(oshape), ProteinInteraction(5))
)

print_model_parameters(model)
train(model, mnist)
'''