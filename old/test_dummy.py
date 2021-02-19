import torch

from torch import nn
from xox import XOXHyperNetwork, DummyHyperNetwork

'''
Test whether dummy hypernetwork works correctly.
'''

net = nn.Linear(2, 2, bias=False)
hyper = DummyHyperNetwork(net)

print("hyper = ", hyper.hyper_weight)
print("net = ", net.weight)

with torch.no_grad(): net.weight.zero_()

hyper.forward()
print("net = ", net.weight)

with torch.no_grad(): net.weight.zero_()
print("net = ", net.weight)

hyper.absorb(1.0)
print("hyper = ", hyper.hyper_weight)



