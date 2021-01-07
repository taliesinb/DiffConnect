import torch

from torch import nn
from data import mnist, cifar

from hyper import RandomBasisHyperNetwork
from train import train, cached_train
from cache import cached

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from param_xox import *
from indent import *
from utils import product

from types import FunctionType

def train_params_and_accuracy(model, iterator, steps, lr, global_seed):
    res = cached_train(model, iterator, max_batches=steps, lr=lr, global_seed=global_seed)
    return {'parameters': res['weight_params'], 'accuracy': res['accuracy']}

cached_train_params_and_accuracy = cached(train_params_and_accuracy)

def mlp(hidden_size=100):
    return nn.Sequential(
        nn.Linear(product(ishape), hidden_size), nn.Tanh(), nn.Linear(hidden_size, oshape)
    )

'''
Naming scheme for the model factories: the function name is built from fragments representing each layer,
separated by _. Each fragment is CamelCase, reflecting the XOX parameterization (with the word Expression dropped).

E.g. RandomGaussian_RandomGaussian = XOXLinear(RandomGaussianExpression(...), RandomGaussianExpression(...))

There should be arguments that carry the unspecified parameters, (e.g. the '...' in the example above). Also, the 
keyword 'genes' should always be present and should be applied to all XOX Linear throughout.
'''

ishape = [28, 28]
oshape = 10

def randomgaussian_learned(genes):
    return XOXLinear(RandomGaussianExpression(ishape), LearnedExpression(oshape), genes)

def randomgaussian_random(genes):
    return XOXLinear(RandomGaussianExpression(ishape), RandomExpression(oshape), genes)

def learnedgaussian_learned(genes):
    return XOXLinear(LearnedGaussianExpression(ishape), LearnedExpression(oshape), genes)

def learnedgaussian_random(genes):
    return XOXLinear(LearnedGaussianExpression(ishape), RandomExpression(oshape), genes)

def random_random(genes):
    return XOXLinear(RandomExpression(ishape), RandomExpression(oshape), genes)

def random_learned(genes):
    return XOXLinear(RandomExpression(ishape), LearnedExpression(oshape), genes)

steps = 10000
lr = 0.001

models = [
    randomgaussian_learned,
    randomgaussian_random,
    learnedgaussian_learned,
    learnedgaussian_random,
    random_random,
    random_learned,
]

def randombasis_mlp(ndims):
    return RandomBasisHyperNetwork(mlp(), ndims=ndims)

for run in range(5):
    with indent:
        print(f"Run number {run}")
        for genes in [5,10,20,30]:#[1,5,10,15,20,25,30]:#range(1,31,5):
            print(f"Training with {genes} genes")
            for model in models:
                with indent:
                    print('Model ', model.__name__)
                    with indent:
                        cached_train_params_and_accuracy((model, {'genes': genes}), mnist, steps=steps, lr=lr, global_seed=run)

        for dims in np.logspace(np.log10(10),np.log10(1000),num = 10,dtype='int'):
            print(f"Training with {dims}-dimensional subspace")
            cached_train_params_and_accuracy((randombasis_mlp, {'ndims': dims}), mnist, steps=steps, lr=lr, global_seed=run)



