import torch

'''
'''

from torch import nn
from data import mnist_generator, cifar_generator
from xox import XOXHyperNetwork, HyperNetwork
from utils import reset_parameters
from train import train

# making a simple MLP
net = nn.Sequential(
    nn.Linear(28*28, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# create a hyper network that produces the weights and biases of the network
hyper = XOXHyperNetwork(net, num_genes=9, fix_gene_matrices=True, fix_o_matrix=True)
reset_parameters(net)
train(net, mnist_generator, 5000, title='fix_both', hyper_net=hyper)

hyper = XOXHyperNetwork(net, num_genes=9, fix_gene_matrices=True, fix_o_matrix=False)
reset_parameters(net)
train(net, mnist_generator, 5000, title='fix_genes', hyper_net=hyper)

hyper = XOXHyperNetwork(net, num_genes=9, fix_gene_matrices=False, fix_o_matrix=True)
reset_parameters(net)
train(net, mnist_generator, 5000, title='fix_o', hyper_net=hyper)
