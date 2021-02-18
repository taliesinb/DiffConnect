import torch
from torch import nn
from data import mnist, cifar

from hyper import RandomBasisHyperNetwork
from train import train, cached_train, train_models
from param_xox import *

from indent import indent
from utils import product, save_to_csv

ishape = (28, 28)
oshape = 10

def make_possibly_shared_expressions(is_shared, fn):
    if is_shared:
        expr1 = expr2 = fn()
    else:
        expr1 = fn()
        expr2 = fn()
    return expr1, expr2

def RandomGaussian_RandomGaussian_RandomGaussian_Random(genes, hshape, is_shared):
    expr1, expr2 = make_possibly_shared_expressions(is_shared, lambda: RandomGaussianExpression(hshape))
    return nn.Sequential(
        XOXLinear(RandomGaussianExpression(ishape), expr1, genes),
        XOXLinear(expr2, RandomExpression(oshape), genes)
    )

def RandomGaussian_LearnedGaussian_LearnedGaussian_Random(genes, hshape, is_shared):
    expr1, expr2 = make_possibly_shared_expressions(is_shared, lambda: LearnedGaussianExpression(hshape))
    return nn.Sequential(
        XOXLinear(RandomGaussianExpression(ishape), expr1, genes),
        XOXLinear(expr2, RandomExpression(oshape), genes)
    )

def LearnedGaussian_LearnedGaussian_LearnedGaussian_Random(genes, hshape, is_shared):
    expr1, expr2 = make_possibly_shared_expressions(is_shared, lambda: LearnedGaussianExpression(hshape))
    return nn.Sequential(
        XOXLinear(LearnedGaussianExpression(ishape), expr1, genes),
        XOXLinear(expr2, RandomExpression(oshape), genes)
    )

def RandomGaussian_LearnedGaussian_LearnedGaussian_Learned(genes, hshape, is_shared):
    expr1, expr2 = make_possibly_shared_expressions(is_shared, lambda: LearnedGaussianExpression(hshape))
    return nn.Sequential(
        XOXLinear(RandomGaussianExpression(ishape), expr1, genes),
        XOXLinear(expr2, LearnedExpression(oshape), genes)
    )

two_layer_models = [
    RandomGaussian_RandomGaussian_RandomGaussian_Random,
    RandomGaussian_LearnedGaussian_LearnedGaussian_Random,
    LearnedGaussian_LearnedGaussian_LearnedGaussian_Random,
    RandomGaussian_LearnedGaussian_LearnedGaussian_Learned
]

genes = range(1,20)
hshape = [[10, 10]]
num_steps = 10
records = train_models(two_layer_models, {'genes': genes, 'hshape':hshape, 'is_shared': [False, True]}, mnist, max_batches=num_steps, fields=['weight_param_count', 'best_accuracy'], runs=10)
save_to_csv('scaling_two_layers.csv', records)
