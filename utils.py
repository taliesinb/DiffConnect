import torch
import torch.nn
import torch.optim
import tensorboardX
import numpy as np

import itertools as it
from functools import reduce
import operator

import pandas

import datetime
import pathlib
import os
import random
import math

'''
this lets us both randomize the order, and take a limited number of elements from a list.
'''
def shuffle_and_subsample(values, shuffle, count):
    if shuffle:
        values = values.copy()
        random.shuffle(values)
    size = len(values)
    if count and count < size:
        values = [values[math.floor(i)] for i in np.arange(0, size, (size - 1) / (count - 1))]
    return values

def rms(arr):
    if type(arr) is list:
        return list(map(rms, arr))
    else:
        return torch.sqrt(torch.mean(arr.detach().pow(2))).item()

def ss(arr):
    if type(arr) is list:
        return list(map(ss, arr))
    else:
        return torch.sum(arr.detach().pow(2)).item()

def cartesian_product(container):
    if isinstance(container, dict):
        keys = list(container.keys())
        values = list(container.values())
        return [dict(zip(keys, entry)) for entry in cartesian_product(values)]
    return list(it.product(*container))

def product(arr):
    return reduce(operator.mul, arr, 1)

def sums(arr):
    return list(it.accumulate(arr, operator.add, initial=0))

def products(arr):
    return list(it.accumulate(arr, operator.mul, initial=1))

def most(arr):
    return arr[:-1]

def rest(arr):
    return arr[1:]

def reverse(arr):
    return arr[::-1]

def tap(f):
    def f2(*args):
        print("-----\n", args)
        return f(*args)
    return f2

def uniform_stride(shape):
    return reverse(products(reverse(rest(shape))))

def subview(array, shape, offset):
    return torch.as_strided(array, shape, uniform_stride(shape), offset)

def split_with_shapes(array, shapes):
    sizes = map(product, shapes)
    return [subview(array, shape, offset) for shape, offset in zip(shapes, sums(sizes))]

def split_as(array, other_arrays):
    return split_with_shapes(array, [other.shape for other in other_arrays])

def to_vector(arrays):
    return torch.cat([a.view(-1) for a in arrays])

# def multi_dot(vec, other_arrays):
#     vecs = split_as(vec, other_arrays)
#     return sum(map(tap(torch.dot), other_arrays, vecs))

def multi_dot(vec, other_arrays):
    return torch.dot(vec, to_vector(other_arrays))

def hessian_vector_product(loss, params, vector):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    jvp = multi_dot(vector, grads)
    hvp = torch.autograd.grad(jvp, params, retain_graph=True)
    return to_vector(hvp)

def cross_entropy_loss(net, batch):
    inputs, labels = batch
    return torch.nn.functional.cross_entropy(net(inputs), labels)

def gaussian_filters(shape, positions, isigma):
    ranges = [torch.linspace(-1.0, 1.0, d) for d in shape]
    coords = torch.cartesian_prod(*ranges)
    dists = torch.cdist(coords, positions)
    if isinstance(isigma, torch.Tensor): 
        isigma = isigma.unsqueeze(0)
    y = torch.exp(-(dists * isigma) ** 2)
    return y.view(*shape, positions.shape[0])

opt_mapping = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD,
    'RMSprop': torch.optim.RMSprop,
    'Adadelta': torch.optim.Adadelta,
    'Adagrad': torch.optim.Adagrad,
    'AdamW': torch.optim.AdamW,
    'Rprop': torch.optim.Rprop,
    'ASGD': torch.optim.ASGD
}

run_count = 0

def make_log_writer(log_dir, title):
    global run_count
    if not log_dir: return None
    if title is None:
        run_count += 1
        title = f"{run_count:03d}"
    return tensorboardX.SummaryWriter(log_dir + '/' + str(title), flush_secs=2)

def test_accuracy(net, generator, max_items=500):
    total = correct_total = 0
    for img, label, *rest in generator:
        label_prime = net(img, *rest).argmax(1)
        correct = sum(label == label_prime).item()
        correct_total += correct
        total += img.shape[0]
        if total > max_items:
            break
    return correct_total / total


def tensor_f32(x):
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], torch.Tensor):
        return torch.stack(x).float()
    return torch.tensor(x, dtype=torch.float32)

def tensor_int(x):
    return torch.tensor(x, dtype=torch.long)


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
    elif isinstance(x, dict):
        return '{' + ','.join([tostr(k) + ':' + tostr(v) for k, v in x.items()]) + '}'
    elif isinstance(x, (np.ndarray, torch.Tensor)):
        return '<' + tostr(x.shape) + '>'
    elif isinstance(x, torch.Size):
        return '⨉'.join(map(str, list(x)))
    return str(x)


def print_row(*args, colsize=15):
    print(' '.join(tostr(arg).rjust(colsize, ' ') for arg in args))


def count_parameters(model):
    return sum(np.prod(p.shape) for p in model.parameters())


def print_tensor_sizes(*args):
    print('\t'.join(map(lambda x: tostr(x.size()), args)))


def print_model_parameters(model):
    print_row('parameter'.rjust(30, ' '), 'shape', 'mean', 'sd', 'rms', 'gradrms', colsize=10)
    for key, value in model.named_parameters():
        key = tostr(key).rjust(30, ' ')
        if value is None: 
            print(key, 'NOT DEFINED')
            continue
        mean = value.mean().item()
        sd = np.sqrt(value.var().item())
        shape = tuple(value.shape)
        print_row(key, shape, mean, sd, rms(value), rms(value.grad) if value.grad is not None else '', colsize=10)

def print_model_gradients(model):
    print_row('gradient', 'RMS')
    for key, value in model.named_parameters():
        if value.requires_grad and value.grad is not None:
            grad = value.grad
            print_row(key, rms(grad))


def batched_to_flat_image(t):
    import torchvision

    if isinstance(t, list):
        t = torch.cat(t)

    shape = t.shape
    n = shape[0]
    rank = len(shape)

    red_blue = True
    if rank == 2:
        w = shape[1]
        if w > 8:
            h = np.ceil(np.sqrt(w))
            w = w // h
        else:
            h = w
            w = 1
        shape = [n, 1, h, w]
    elif rank == 3:
        shape = [n, 1, shape[1], shape[2]]
    elif rank == 4:
        shape = shape
        red_blue = False
    t = t.view(*shape)

    t_min = t.min()
    t_max = t.max()
    if red_blue and t_min < 0 < t_max:
        scale = max(-t_min, t_max)
        # for positive, shift the         green and blue down
        # for negative, shift the red and green          down
        scaled = t / scale
        r = 1 + torch.clamp(scaled, -1, 0)
        g = 1 - abs(scaled)
        b = 1 - torch.clamp(scaled, 0, 1)
        t = torch.cat([r, g, b], dim=1)

    grid = torchvision.utils.make_grid(t, 10, normalize=(not red_blue), padding=1)
    return grid


def print_image(t):
    from matplotlib import pyplot
    rgb = batched_to_flat_image(t)
    rgb = np.transpose(rgb, [1, 2, 0])
    pyplot.imshow(1 - rgb)
    pyplot.show()

def is_factory(t):
    return isinstance(t, tuple) and len(t) >= 1 and callable(t[0])

def run_factory(t):
    factory_fn = t[0]
    if len(t) == 1:
        args, kwargs = [], {}
    if len(t) == 2 and isinstance(t[1], tuple):
        args, kwargs = t[1], {}
    elif len(t) == 2 and isinstance(t[1], dict):
        args, kwargs = [], t[1]
    elif len(t) == 3 and isinstance(t[1], tuple) and isinstance(t[2], dict):
        args, kwargs = t
    return factory_fn(*args, **kwargs)

def backup_file(path):
    path_obj = pathlib.Path(path)
    if not path_obj.exists():
        return
    mtime = datetime.datetime.fromtimestamp(path_obj.stat().st_mtime)
    time_str = mtime.strftime("%Yy%mm%dd%Hh%Mm")
    backup_path = path + '.' + time_str + '.backup'
    os.rename(path, backup_path)
    return backup_path

def save_to_csv(path, records, exclude=[], backup_previous_file=True):
    if backup_previous_file: backup_file(path)
    return pandas.DataFrame.from_records(records, exclude=exclude).to_csv(path)