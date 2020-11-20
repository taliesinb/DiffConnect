from torch import nn
from xox import XOXHyperNetwork, HyperNetwork
from utils import *

'''
Test whether hypernetwork is initialized in such a way that the generated weights and biases 
roughly match the corresponding initialization torch would have used
'''

net = nn.Sequential(
    nn.Linear(1, 100, bias=True),
    nn.Linear(100, 100, bias=True),
    nn.Linear(100, 1, bias=False),
)

hyper = XOXHyperNetwork(net, num_genes=100)

print_model_parameters(net)
hyper.forward()
print_model_parameters(net)