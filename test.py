import torch

from torch import nn
from data import mnist_generator, cifar_generator
from xox import XOXHyperNetwork, XOXLinear

from train import train

hyper = XOXHyperNetwork(10, sharing=True)
net = nn.Sequential(
    hyper.linear(28*28, 64),
    nn.ReLU(),
    hyper.linear(64, 10)
)

train(net, mnist_generator, 5000, title='ordinary')

hyper = XOXHyperNetwork(10, sharing=True)
net_2d = nn.Sequential(
    hyper.linear_2d(28*28, 64),
    nn.ReLU(),
    hyper.linear(64, 10)
)


train(net_2d, mnist_generator, 5000, title='2d')
