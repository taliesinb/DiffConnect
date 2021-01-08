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

def RandomGaussian_Learned(genes):
    return XOXLinear(RandomGaussianExpression(ishape), LearnedExpression(oshape), genes)

def RandomGaussian_Random(genes):
    return XOXLinear(RandomGaussianExpression(ishape), RandomExpression(oshape), genes)

def LearnedGaussian_Learned(genes):
    return XOXLinear(LearnedGaussianExpression(ishape), LearnedExpression(oshape), genes)

def LearnedGaussian_Random(genes):
    return XOXLinear(LearnedGaussianExpression(ishape), RandomExpression(oshape), genes)

def Random_Random(genes):
    return XOXLinear(RandomExpression(ishape), RandomExpression(oshape), genes)

def Random_Learned(genes):
    return XOXLinear(RandomExpression(ishape), LearnedExpression(oshape), genes)

one_layer_models = [
    RandomGaussian_Learned,
    RandomGaussian_Random,
    LearnedGaussian_Learned,
    LearnedGaussian_Random,
    Random_Random,
    Random_Learned,
]

records = train_models(one_layer_models, {'genes': [5]}, mnist, max_batches=5, fields=['weight_param_count', 'best_accuracy'])
save_to_csv('one_layer_models.csv', records)
