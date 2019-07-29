import torch

from torch import nn
from data import mnist_generator, cifar_generator
from xox import XOXHyperNetwork, HyperNetwork
from utils import reset_parameters
from train import train

net = nn.Sequential(
    nn.Linear(28*28, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

train(net, mnist_generator, 2000, title='ordinary')
# should achieve around 95% final accuracy

reset_parameters(net)

hyper = XOXHyperNetwork(net, num_genes=9)

train(net, mnist_generator, 2000, title='hyper', hyper_net=hyper)
# should achieve around 90% final accuracy
