import torch

from torch import nn
from data import mnist, cifar

from hyper import RandomBasisHyperNetwork
from train import train
from cache import cached, load_cached_results_as_pandas

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from param_xox import *
from indent import *
from utils import product

@cached
def train_mnist_multi_layer_normal(steps=5000):
    net = nn.Sequential(
        nn.Linear(28*28, 100),
        nn.Tanh(),
        nn.Linear(100,10)
        )
    return train(net, mnist, max_batches=steps, title='mnist_multi', log_dir=None)

@cached
def Spatial_to_Spatial_to_Spatial_multi_layer_xox(num_genes, steps=5000):
    net = nn.Sequential(
        ParameterizedXOXLinear(GaussianExpression([28, 28]), GaussianExpression([10,10]), num_genes),
        nn.Tanh(),
        ParameterizedXOXLinear(GaussianExpression([10, 10]), GaussianExpression([2,5]), num_genes)
    )
    return train(net, mnist, max_batches=steps, title='gaussian_to_gaussian_to_gaussian', log_dir=None)

@cached
def Spatial_to_Spatial_to_Learned_multi_layer_xox(num_genes, steps=5000):
    net = nn.Sequential(
        ParameterizedXOXLinear(GaussianExpression([28, 28]), GaussianExpression([10,10]), num_genes),
        nn.Tanh(),
        ParameterizedXOXLinear(GaussianExpression([10, 10]), LearnedExpression(10), num_genes)
    )
    return train(net, mnist, max_batches=steps,title='gaussian_to_gaussian_to_learned', log_dir=None)

@cached
def Random_to_Random_to_Random_multi_layer_xox(num_genes, steps=5000):
    net = nn.Sequential(
        ParameterizedXOXLinear(RandomExpression(28*28), RandomExpression(10*10), num_genes),
        nn.Tanh(),
        ParameterizedXOXLinear(RandomExpression(10*10), RandomExpression(10), num_genes)
    )
    return train(net, mnist, max_batches=steps,title='random_to_random_to_random', log_dir=None)

@cached
def Learned_to_Learned_to_Learned_multi_layer_xox(num_genes, steps=5000):
    net = nn.Sequential(
        ParameterizedXOXLinear(LearnedExpression(28*28), LearnedExpression(10*10), num_genes),
        nn.Tanh(),
        ParameterizedXOXLinear(LearnedExpression(10*10), LearnedExpression(10), num_genes)
    )
    return train(net, mnist, max_batches=steps,title='learned_to_learned_to_learned', log_dir=None)

@cached
def train_mnist_multi_layer_random_basis(ndims, steps=5000):
    net = nn.Sequential(
        nn.Linear(28*28, 100),
        nn.Tanh(),
        nn.Linear(100,10)
        )
    hyper = RandomBasisHyperNetwork(net, ndims=ndims)
    return train(hyper, mnist, max_batches=steps, title='random_multi', log_dir=None)


print('Loading data')
ranges = 5

for i in [5,10,20,30,40,50]:
    print(f"Training Spatial to Spatial to Spatial with {i} genes")
    for j in range(ranges):
        with indent:
            print(f"Run number {j}")
            Spatial_to_Spatial_to_Spatial_multi_layer_xox(num_genes = i,global_seed = j)

for i in [5,10,20,30,40,50]:
    print(f"Training Spatial_to_Spatial_to_Learned_multi_layer_xox with {i} genes")
    for j in range(ranges):
        with indent:
            print(f"Run number {j}")
            Spatial_to_Spatial_to_Learned_multi_layer_xox(num_genes = i,global_seed = j)


for i in [5,10,20,30,40,50]:
    print(f"Training Random_to_Random_to_Random_multi_layer_xox with {i} genes")
    for j in range(ranges):
        with indent:
            print(f"Run number {j}")
            Random_to_Random_to_Random_multi_layer_xox(num_genes = i,global_seed = j)


for i in [5,10,20,30,40,50]:
    print(f"Training Learned_to_Learned_to_Learned_multi_layer_xox with {i} genes")
    for j in range(ranges):
        with indent:
            print(f"Run number {j}")
            Learned_to_Learned_to_Learned_multi_layer_xox(num_genes = i,global_seed = j)

for i in [10, 50, 100, 200, 500, 1000]:
    print(f"Training with {i}-dimensional subspace")
    for j in range(ranges):
        with indent:
            print(f"Run number {j}")
            train_mnist_multi_layer_random_basis(i, global_seed=j)

print('Done')

results_normal = train_mnist_multi_layer_normal()

results_spatial_to_spatial_to_spatial = load_cached_results_as_pandas(Spatial_to_Spatial_to_Spatial_multi_layer_xox)
results_spatial_to_spatial_to_spatial['label'] = 'results_spatial_to_spatial_to_spatial'

results_spatial_to_spatial_to_learned = load_cached_results_as_pandas(Spatial_to_Spatial_to_Learned_multi_layer_xox)
results_spatial_to_spatial_to_learned['label'] = 'results_spatial_to_spatial_to_learned'

results_random_to_random_to_random = load_cached_results_as_pandas(Random_to_Random_to_Random_multi_layer_xox)
results_random_to_random_to_random['label'] = 'results_random_to_random_to_random'

results_learned_to_learned_to_learned = load_cached_results_as_pandas(Learned_to_Learned_to_Learned_multi_layer_xox)
results_learned_to_learned_to_learned['label'] = 'results_learned_to_learned_to_learned'

results_random_basis = load_cached_results_as_pandas(train_mnist_multi_layer_random_basis)
results_random_basis['label'] = 'results_random_basis'

results = pd.concat([results_spatial_to_spatial_to_spatial,results_spatial_to_spatial_to_learned,results_random_to_random_to_random,results_random_basis,results_learned_to_learned_to_learned])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.axhline(y=results_normal['accuracy'], color='black', linewidth=2, alpha=.7)
sns.pointplot(x='weight_params', y='accuracy', data=results, aspect=1.4, hue='label')

plt.show()