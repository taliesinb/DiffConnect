from torch import nn
from data import mnist_generator, cifar_generator
from xox import XOXHyperNetwork, XOXLinear

from train import train


def random_train(num_genes, freeze=False):
    print(f"num genes = {num_genes}")
    hyper = XOXHyperNetwork(num_genes, freeze)
    net = nn.Sequential(
        XOXLinear(28*28, 50, hyper=hyper),
        nn.ReLU(),
        XOXLinear(50, 10, hyper=hyper)
    )
    name = f"{num_genes:02d}"
    if freeze: name = "frozen/" + name
    stats = train(net, mnist_generator, 50000, title=name, log_weights=True)
    return net, stats

net = nn.Sequential(
    nn.Linear(28*28, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

#random_train(64, False)
#random_train(8, False)

train(net, mnist_generator, 10000, title="baseline")

for num_genes in range(1, 21, 1):
    random_train(num_genes, False)
    random_train(num_genes, True)

#train(net, mnist_generator, 25000, title="baseline")

#for num_genes in range(2, 10, 2):
#    random_train(num_genes)
