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

def RandomGaussian_RandomGaussian_RandomGaussian_Random(genes, hshape):
    return nn.Sequential(
        XOXLinear(RandomGaussianExpression(ishape), RandomGaussianExpression(hshape), genes),
        XOXLinear(RandomGaussianExpression(hshape), RandomExpression(oshape), genes)
    )

def RandomGaussian_RandomGaussian_LearnedGaussian_Random(genes, hshape):
    return nn.Sequential(
        XOXLinear(RandomGaussianExpression(ishape), RandomGaussianExpression(hshape), genes),
        XOXLinear(LearnedGaussianExpression(hshape), RandomExpression(oshape), genes)
    )

def RandomGaussian_LearnedGaussian_LearnedGaussian_Random(genes, hshape):
    return nn.Sequential(
        XOXLinear(RandomGaussianExpression(ishape), LearnedGaussianExpression(hshape), genes),
        XOXLinear(LearnedGaussianExpression(hshape), RandomExpression(oshape), genes)
    )

def LearnedGaussian_LearnedGaussian_LearnedGaussian_Random(genes, hshape):
    return nn.Sequential(
        XOXLinear(LearnedGaussianExpression(ishape), LearnedGaussianExpression(hshape), genes),
        XOXLinear(LearnedGaussianExpression(hshape), RandomExpression(oshape), genes)
    )

two_layer_models = [
    RandomGaussian_RandomGaussian_RandomGaussian_Random,
    RandomGaussian_RandomGaussian_LearnedGaussian_Random,
    RandomGaussian_LearnedGaussian_LearnedGaussian_Random,
    LearnedGaussian_LearnedGaussian_LearnedGaussian_Random
]

genes = [1,5,10] # range(1,10)
hshape = [[10, 10], [7, 7], [5, 5], [3, 3], [2, 2]]
records = train_models(two_layer_models, {'genes': genes, 'hshape':hshape}, mnist, max_batches=5, fields=['weight_param_count', 'best_accuracy'])
save_to_csv('scaling_two_layers.csv', records)
