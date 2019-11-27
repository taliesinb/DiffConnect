import torch
import torch.nn
import numpy as np


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
    else:
        return str(x)


def print_row(*args):
    print(' '.join(tostr(arg).rjust(12, ' ') for arg in args))


def count_parameters(model):
    return sum(np.prod(p.shape) for p in model.parameters())


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
