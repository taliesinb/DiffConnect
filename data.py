import torch
import torchvision
import itertools
import utils

def _generic_iterator(is_training, factory, batch_size=64, flatten=True):
    trans = torchvision.transforms.ToTensor()
    if flatten:
        trans = torchvision.transforms.Compose([trans, torch.flatten])
    dataset = factory('data/' + factory.__name__.lower(), train=is_training, download=True, transform=trans)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return itertools.cycle(dataset_loader)

def _generic_iterator_pair(**kw_args):
    return _generic_iterator(True, **kw_args), _generic_iterator(False, **kw_args)

def cifar(**kw_args):
    return _generic_iterator_pair(factory=torchvision.datasets.CIFAR10, **kw_args)

def mnist(**kw_args):
    return _generic_iterator_pair(factory=torchvision.datasets.MNIST, **kw_args)
