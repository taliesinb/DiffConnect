import torch

from torch import nn
from data import mnist_generator, cifar_generator
from xox import XOXHyperNetwork, HyperNetwork

from train import train
from cache import cached, load_cached_results_as_pandas

def train_mnist_multi_layer():
    net = nn.Sequential(
        nn.Linear(28 * 28, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

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


import seaborn as sns
# import pandas as pd
import matplotlib.pyplot as plt

print('loading data')
for i in range(1, 10):
    train_mnist_single_layer_xox(i)
print('done')

results_xox = load_cached_results_as_pandas(train_mnist_single_layer_xox)
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


plt.show()