import torch

from torch.nn import functional, Module

import numpy as np
import tensorboardX
import torchvision


run_count = 0


def train(net, generator_factory, max_batches, *,
          lr=0.01, log_dir='runs', title=None, batch_size=64, hyper_net=None, params=None):
    global run_count
    if params is None:
        if hyper_net:
            params = hyper_net.parameters()
        else:
            params = net.parameters()
    params = list(params)
    weight_shapes = [tuple(p.shape) for p in params]
    weight_params = sum(np.prod(shape) for shape in weight_shapes)
    print(f"Training {len(params)} weights with total {weight_params} parameters:")
    print(weight_shapes)
    if log_dir:
        if title is None:
            run_count += 1
            title = f"{run_count:03d}"
        writer = tensorboardX.SummaryWriter(log_dir + '/' + str(title), flush_secs=2)
    else:
        writer = None
    opt = torch.optim.Adam(params, lr=lr)
    time = 0
    accuracy_generator = generator_factory(batch_size, is_training=True)
    training_generator = generator_factory(batch_size, is_training=False)
    running_loss = None
    loss_history = []
    acc_history = []
    for img, label in training_generator:
        time += 1
        opt.zero_grad()
        img = torch.flatten(img, start_dim=1)
        if hyper_net:
            net.zero_grad()
            hyper_net.zero_grad()
            hyper_net.forward()
        label_prime = net(img)
        loss = functional.cross_entropy(label_prime, label)
        loss.backward()
        if hyper_net:
            hyper_net.backward()
        opt.step()
        loss = loss.item()
        loss_history.append(loss)
        if running_loss is None:
            running_loss = loss
        else:
            running_loss = 0.95 * running_loss + 0.05 * loss
        if time % 10 == 0 and writer:
            writer.add_scalar("loss", running_loss, time)
        if time % 500 == 0:
            if time % 2500 == 0:
                acc = test_accuracy(net, accuracy_generator)
                acc_history.append((time, acc))
                if writer: writer.add_scalar("accuracy", acc, time)
                print(f"{time:>5d}\t{running_loss:.3f}\t{acc:.3f}")
            else:
                print(f"{time:>5d}\t{running_loss:.3f}")
        if time > max_batches:
            break
    acc = test_accuracy(net, accuracy_generator, max_batches=10000)
    if writer:
        writer.add_scalar("accuracy", acc, time)
        writer.close()
    print(f"final accuracy = {acc:.3f}")
    history = {'loss': loss_history, 'accuracy': acc_history}
    return {'loss': running_loss, 'accuracy': acc, 'history': history,
            'weight_shapes': weight_shapes, 'weight_params': weight_params,
            'batch_size': batch_size, 'batches': max_batches}


def test_accuracy(net, generator, max_batches=5000):
    total = correct_total = 0
    for img, label in generator:
        img = torch.flatten(img, start_dim=1)
        label_prime = net(img).argmax(1)
        correct = sum(label == label_prime).item()
        correct_total += correct
        total += img.shape[0]
        if total > max_batches:
            break
    return correct_total / total


'''
def visualize_weights(layer, n_colors=1):
    with torch.no_grad():
        if isinstance(layer, torch.nn.Linear):
            weight = layer.weight
        else:
            weight, _ = layer.calculate_weight()
            # ^ for XOXLinear and friends
        output_size, input_size = weight.shape
        input_height = input_width = int(np.sqrt(input_size))
        size = (output_size, n_colors, input_height, input_width)
        weight_reshaped = weight.reshape(size).clone()
        img = batched_to_flat_image(weight_reshaped)
        return img

'''
def batched_to_flat_image(t):
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
        shape = list(shape)
        if shape[1] > 1:
            red_blue = False
    t = t.view(*shape)

    t_min = t.min()
    t_max = t.max()
    if red_blue and t_min < 0 < t_max:
        #sorted, _ = t.flatten().sort()
        #n = sorted.numel()
        #t_min = sorted[int(n / 5)]
        #t_max = sorted[int(n * 4 / 5)]
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
