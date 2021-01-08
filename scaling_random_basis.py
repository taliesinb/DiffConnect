import torch
from torch import nn
from data import mnist, cifar

from hyper import RandomBasisHyperNetwork
from train import train_models

from indent import indent
from utils import product, save_to_csv

import numpy as np

ishape = (28, 28)
oshape = 10

#, 'hshape': [[5,20], [10, 10], [20, 5]]
def mlp(hidden_size=100):
    return nn.Sequential(
        nn.Linear(product(ishape), hidden_size), nn.Tanh(), nn.Linear(hidden_size, oshape)
    )

def slp():
    return nn.Linear(product(ishape), oshape)

def randombasis_mlp(ndims):
    return RandomBasisHyperNetwork(mlp(), ndims=ndims)

def randombasis_slp(ndims):
    return RandomBasisHyperNetwork(slp(), ndims=ndims)

random_basis_models = [
    randombasis_slp,
    randombasis_mlp
]

ndims = list(np.logspace(np.log10(10),np.log10(1000), num=10, dtype='int'))
records = train_models(random_basis_models, {'ndims': ndims}, mnist, max_batches=40000, test_interval=2000, fields=['weight_param_count', 'best_accuracy'])
save_to_csv('scaling_random_basis.csv', records)

