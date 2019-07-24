import torch
import torchvision
import itertools


def cifar_generator(batch_size, is_training=True):
    trans = torchvision.transforms.ToTensor()
    cifar = torchvision.datasets.CIFAR10('/tmp/cifar10', train=is_training, download=True, transform=trans)
    cifar_loader = torch.utils.data.DataLoader(cifar, batch_size=batch_size, shuffle=True)
    return itertools.cycle(cifar_loader)


def mnist_generator(batch_size, is_training=True):
    trans = torchvision.transforms.ToTensor()
    mnist = torchvision.datasets.MNIST('/tmp/mnist', train=is_training, download=True, transform=trans)
    mnist_loader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)
    return itertools.cycle(mnist_loader)
