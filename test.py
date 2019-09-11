import torch

'''
This file demonstrates training of a single MLP to solve MNIST, then followed
by training of a hypernetwork-controlled MLP on the same task. Accuracy is lower for
the hypernetwork approach (90% instead of 95%), but still reasonable.

Output should look something like this:

Training 4 weights with total 50890 parameters:
[(64, 784), (64,), (10, 64), (10,)]
  500	0.108
 1000	0.064
 1500	0.040
 2000	0.030
final accuracy = 0.945
Hypernetwork: 50890 -> 8346
Training 5 weights with total 8346 parameters:
[(784, 9), (64, 9), (64,), (10, 64), (10,)]
  500	0.275
 1000	0.191
 1500	0.156
 2000	0.140
final accuracy = 0.906
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

# train the MLP on mnist for 2000 batches (this should achieve around 95% final accuracy)
# train(net, mnist_generator, 2000, title='ordinary')

# reset the weights and biases in the net to random values
reset_parameters(net)

# create a hyper network that produces the weights and biases of the network
hyper = XOXHyperNetwork(net, num_genes=9, fix_gene_matrices=True, fix_o_matrix=True)

# train the network via the hypernetwork (this should achieve around 90% final accuracy)
train(net, mnist_generator, 2000, title='hyper', hyper_net=hyper)

