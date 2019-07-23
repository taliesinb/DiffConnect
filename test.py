import torch

from torch import nn
from torch.nn import functional, Parameter

from data import mnist_generator, cifar_generator

from xox import XOXHyperNetwork, XOXLinear

import tensorboardX


acc_gen = mnist_generator(5, is_train=False)
def test_accuracy(net):
    total = correct_total = 0
    for img, label in acc_gen:
        img = torch.flatten(img, start_dim=1)
        label_prime = net(img).argmax(1)
        correct = sum(label == label_prime).item()
        correct_total += correct
        total += img.shape[0]
        if total > 1000: break
    return correct_total / total


hyper = XOXHyperNetwork(10)

big = nn.Sequential(
    nn.Linear(28*28, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

layer = XOXLinear(28*28, 50, hyper=hyper)
big_xox = nn.Sequential(
    layer,
    nn.ReLU(),
    XOXLinear(50, 10, prev=layer)
)

small = nn.Linear(28*28, 10)
small_xox = XOXLinear(28*28, 10, hyper=hyper)

net = big_xox

params = list(net.parameters())

num_params = sum(p.numel() for p in params)

print(f"Number of parameters = {num_params}")

writer = tensorboardX.SummaryWriter(log_dir="tb_log")

opt = torch.optim.Adam(params, lr=0.01)
time = 0
for img, label in mnist_generator(64):
    time += 1
    opt.zero_grad()
    img = torch.flatten(img, start_dim=1)
    label_prime = net(img)
    loss = functional.cross_entropy(label_prime, label)
    loss.backward()
    opt.step()
    writer.add_scalar("loss", loss, time)
    if time % 500 == 0:
        if time % 1000 == 0:
            acc = test_accuracy(net)
            writer.add_scalar("accuracy", acc, time)
            print(f"{time:>5d}\t{loss:.3f}\t{acc:.3f}")
        else:
            print(f"{time:>5d}\t{loss:.3f}")
    if time > 5000: break

writer.close()