import torch

from torch import nn
from data import mnist_generator, cifar_generator
from xox import XOXHyperNetwork
from hyper import RandomBasisHyperNetwork

from train import train
from cache import cached, load_cached_results_as_pandas

import seaborn as sns
# import pandas as pd
import matplotlib.pyplot as plt

from indent import *


'''
This file computes the relationship between number of genes in a hypernetwork
and the resulting accuracy on MNIST (after a fixed number of SGD steps).

The idea is that with too few genes, there is not enough capacity in the hypernetwork
to represent a high-performing network. But with more genes the capacity and hence final 
accuracy improves, and eventually we reach diminishing returns.

The underlying network is a single layer perceptron. 

Because each training run takes a while, and we do many training runs (one for each
number of genes), we use the @cached utility to store the results of the training run on
disk so that we can re-run the program, and it will pick up where it dropped off without
recalculating previously computed training runs.

See cache.py for more info.

Output from this script should look something like:

Loading data
Training with 1 genes

Hypernetwork: 7850 -> 804
Training 3 weights with total 804 parameters:
[(784, 1), (10, 1), (10,)]
  500	1.597
        ...
 5000	1.489	0.407
final accuracy = 0.407
Hypernetwork: 7850 -> 804
Training 3 weights with total 804 parameters:
[(784, 1), (10, 1), (10,)]
  500	1.621
        ...
        
[ 5 of these in total ]

Training with 2 genes
        ...

Training with 3 genes
        ...
        
...

Training with 10 genes
'''

@cached
def train_mnist_single_layer_xox(num_genes, steps=5000):
    net = nn.Linear(28*28, 10)
    hyper = XOXHyperNetwork(net, num_genes=num_genes, skip_small=False, skip_vectors=True)
    return train(net, mnist_generator, steps, title='hyper', hyper_net=hyper, log_dir=None)

@cached
def train_mnist_single_layer_ha(num_genes, steps=5000):
    raise NotImplementedError()

@cached
def train_mnist_single_layer_normal(steps=5000):
    net = nn.Linear(28*28, 10)
    return train(net, mnist_generator, steps, log_dir=None)

@cached
def train_mnist_single_layer_random_basis(ndims, steps=5000):
    net = nn.Linear(28*28, 10)
    hyper = RandomBasisHyperNetwork(net, ndims=ndims)
    return train(net, mnist_generator, steps, title='random', hyper_net=hyper, log_dir=None)


print('Loading data')

for i in range(1, 10):
    print(f"Training with {i} genes")
    for j in range(5):
        with indent:
            print(f"Run number {j}")
            train_mnist_single_layer_xox(i, global_seed=j)

for i in [10, 50, 100, 200, 500]:
    print(f"Training with {i}-dimensional subspace")
    for j in range(5):
        with indent:
            print(f"Run number {j}")
            train_mnist_single_layer_random_basis(i, global_seed=j)

print('Done')

results_xox = load_cached_results_as_pandas(train_mnist_single_layer_xox)
results_random = load_cached_results_as_pandas(train_mnist_single_layer_random_basis)
results_normal = train_mnist_single_layer_normal()

print(results_xox)

full_params = results_normal['weight_params']
results_xox['capacity'] = results_xox['weight_params'].apply(lambda x: x / full_params)

fig = plt.figure()

ax1 = fig.add_subplot(111)
sns.pointplot(x='num_genes', y='accuracy', data=results_xox, aspect=1.4, ax=ax1, join=False)
ax1.axhline(y=results_normal['accuracy'], color='black', linewidth=2, alpha=.7)

ax2 = plt.twinx(ax1)
plt2 = sns.pointplot(x='num_genes', y='capacity', data=results_xox, aspect=1.4, ax=ax2, join=False, color="#bb3f3f")

fig = plt.figure()
ax1 = fig.add_subplot(111)
sns.pointplot(x='ndims', y='accuracy', data=results_random)

plt.show()