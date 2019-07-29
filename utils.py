import torch
import torch.nn
import numpy as np


def reset_parameters(model):
    model.apply(_reset_parameters)

def _reset_parameters(layer):
    if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
        layer.reset_parameters()


def tostr(x):
    if isinstance(x, float):
        return f'{x:.4f}'
    elif isinstance(x, (list, tuple)):
        return ','.join(map(tostr, x))
    else:
        return str(x)


def print_row(*args):
    print(' '.join(tostr(arg).rjust(12, ' ') for arg in args))


def summarize_parameters(model):
    print_row('parameter', 'shape', 'mean', 'sd')
    for key, value in model.named_parameters():
        mean = value.mean().item()
        sd = np.sqrt(value.var().item())
        shape = tuple(value.shape)
        print_row(key, shape, mean, sd)


def summarize_gradient(model):
    print_row('gradient', 'mean', 'sd')
    for key, value in model.named_parameters():
        if value.grad:
            mean = value.mean().item()
            sd = np.sqrt(value.var().item())
            print_row(key, mean, sd)
        print(key)
