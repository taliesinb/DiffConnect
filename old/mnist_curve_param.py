import torch

from torch import nn
from data import mnist, cifar

from hyper import RandomBasisHyperNetwork
from train import train
from cache import cached, load_cached_results_as_pandas

import numpy as np

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from param_xox import *
from indent import *
from utils import product

tot_steps = 10000
lr_overall = 0.001

@cached
def train_mnist_single_layer_gaussian_to_learned(num_genes, steps=tot_steps):
    net = ParameterizedXOXLinear(GaussianExpression([28, 28]), LearnedExpression(10), num_genes)
    return train(net, mnist, max_batches=steps, title='gaussian_to_learned', log_dir=None, lr = lr_overall)

@cached
def train_mnist_single_layer_learned_gaussian_to_learned(num_genes, steps=tot_steps):
    net = ParameterizedXOXLinear(LearnedGaussianExpression([28, 28]), LearnedExpression(10), num_genes)
    return train(net, mnist, max_batches=steps, title='learned_gaussian_to_learned', log_dir=None, lr = lr_overall)

@cached
def train_mnist_single_layer_learned_gaussian_to_random(num_genes, steps=tot_steps):
    net = ParameterizedXOXLinear(LearnedGaussianExpression([28, 28]), RandomExpression(10), num_genes)
    return train(net, mnist, max_batches=steps, title='learned_gaussian_to_random', log_dir=None, lr = lr_overall)

@cached
def train_mnist_single_layer_gaussian_to_random(num_genes, steps=tot_steps):
    net = ParameterizedXOXLinear(GaussianExpression([28, 28]), RandomExpression(10), num_genes)
    return train(net, mnist, max_batches=steps, title='gaussian_to_random', log_dir=None, lr = lr_overall)

@cached
def train_mnist_single_layer_random_to_random(num_genes, steps=tot_steps):
    net = ParameterizedXOXLinear(RandomExpression(28 * 28), RandomExpression(10), num_genes)
    return train(net, mnist, max_batches=steps, title='random_to_random', log_dir=None, lr = lr_overall)

@cached
def train_mnist_single_layer_normal(steps=tot_steps):
    net = nn.Linear(28*28, 10)
    return train(net, mnist, max_batches=steps, log_dir=None, lr = lr_overall)

@cached
def train_mnist_single_layer_random_basis(ndims, steps=tot_steps):
    net = nn.Linear(28*28, 10)
    hyper = RandomBasisHyperNetwork(net, ndims=ndims)
    return train(hyper, mnist, max_batches=steps, title='random', log_dir=None, lr = lr_overall)

# @cached
# def train_mnist_single_layer_random_basis_no_bias(ndims, steps=tot_steps):
#     net = nn.Linear(28*28, 10)
#     hyper = RandomBasisNoBiasHyperNetwork(net, ndims=ndims)
#     return train(hyper, mnist, max_batches=steps, title='random', log_dir=None, lr = lr_overall)

print('Loading data')

for i in range(1,30):
    print(f"Training with {i} genes")
    for j in range(2):
        with indent:
            print(f"Run number {j}")
            #train_mnist_single_layer_gaussian_to_learned(i, global_seed=j)
            #train_mnist_single_layer_learned_gaussian_to_learned(i, global_seed=j)
            train_mnist_single_layer_gaussian_to_random(i, global_seed=j)
            # train_mnist_single_layer_random_to_random(i, global_seed=j)
            train_mnist_single_layer_learned_gaussian_to_random(i,global_seed=j)


for i in np.logspace(np.log10(10),np.log10(1000),num = 5,dtype='int'):
    print(f"Training with {i}-dimensional subspace")
    for j in range(1):
        with indent:
            print(f"Run number {j}")
            train_mnist_single_layer_random_basis(i, global_seed=j)
            # train_mnist_single_layer_random_basis_no_bias(i, global_seed=j)

print('Done')


# results_gaussian_to_learned = load_cached_results_as_pandas(train_mnist_single_layer_gaussian_to_learned)
 # results_gaussian_to_learned['label'] = 'gaussian to learned'

# results_learned_gaussian_to_learned = load_cached_results_as_pandas(train_mnist_single_layer_learned_gaussian_to_learned)
# results_learned_gaussian_to_learned['label'] = 'learned gaussian to learned'

results_learned_gaussian_to_random = load_cached_results_as_pandas(train_mnist_single_layer_learned_gaussian_to_random)
results_learned_gaussian_to_random['label'] = 'learned gaussian to random'

results_gaussian_to_random = load_cached_results_as_pandas(train_mnist_single_layer_gaussian_to_random)
results_gaussian_to_random['label'] = 'gaussian to random'

# results_random_to_random = load_cached_results_as_pandas(train_mnist_single_layer_random_to_random)
# results_random_to_random['label'] = 'random to random'

results_random_basis = load_cached_results_as_pandas(train_mnist_single_layer_random_basis)
results_random_basis['label'] = 'random basis'

results_normal = train_mnist_single_layer_normal()

results = pd.concat([
    results_learned_gaussian_to_random,   # !
    #results_learned_gaussian_to_learned,  # x
    results_gaussian_to_random,           # !
    #results_gaussian_to_learned,          # x
    results_random_basis,                 # !
    #results_random_to_random              # x
    ])

results.to_csv("mnist_curve.csv", columns=['label', 'weight_params', 'accuracy'])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.axhline(y=results_normal['accuracy'], color='black', linewidth=2, alpha=.7)
sns.pointplot(x='weight_params', y='accuracy', data=results, aspect=1.4, hue='label')

plt.show()
