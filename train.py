import torch

from indent import *
from torch.nn import functional, Module

import numpy as np
import tensorboardX
import torchvision


run_count = 0

opt_mapping = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD,
    'RMSprop': torch.optim.RMSprop
}

@indenting
def train(net, generator_factory, max_batches, *,
          optimizer='Adam', lr=0.01,
          log_dir='runs', title=None, batch_size=64,
          hyper_net=None, params=None, flatten=True):
    global run_count

    if params is None:
        # if a hypernetwork was provided, we don't directly train via the network parameters,
        # we train via the hypernetwork parameters (which produce the network parameters)
        if hyper_net:
            params = hyper_net.parameters()
        else:
            params = net.parameters()

    params = list(params)
    weight_shapes = [tuple(p.shape) for p in params]
    weight_params = sum(np.prod(shape) for shape in weight_shapes)
    title_str = f'Training "{title}"' if title else 'Training'
    shape_str = ' '.join('×'.join(map(str, x)) for x in weight_shapes)
    print(f"{title_str} (arrays: {len(params)}, params: {weight_params}, shapes: {shape_str})")
    if log_dir:
        if title is None:
            run_count += 1
            title = f"{run_count:03d}"
        writer = tensorboardX.SummaryWriter(log_dir + '/' + str(title), flush_secs=2)
    else:
        writer = None

    opt = opt_mapping[optimizer](params, lr=lr)

    time = 0
    accuracy_generator = generator_factory(batch_size, is_training=True)
    training_generator = generator_factory(batch_size, is_training=False)
    running_loss = None
    loss_history = []
    acc_history = []

    # main training loop
    for inputs, labels, *rest in training_generator:
        time += 1
        opt.zero_grad()
        if flatten:
            inputs = torch.flatten(inputs, start_dim=1)

        # if we have a hyper network, we should use its .forward method to
        # derive the parameters for our network
        if hyper_net:
            net.zero_grad()
            hyper_net.zero_grad()
            hyper_net.forward()

        # apply the net to the input batch
        res = net(inputs, *rest)
        if not isinstance(res, tuple):
            res = res, 0

        # the second returned value should be additional losses (if any)
        labels_prime, extra_loss = res

        # calculate the loss
        loss = functional.cross_entropy(labels_prime, labels)
        loss += extra_loss

        # obtain gradients of the loss
        loss.backward()

        # if we have a hypernetwork, we need to propogate those gradients
        # from the network back into the hypernetwork
        if hyper_net:
            hyper_net.backward()

        # do one step of optimization
        opt.step()

        # report losses, etc.
        loss = loss.item()
        loss_history.append(loss)
        if running_loss is None:
            running_loss = loss
        else:
            running_loss = 0.95 * running_loss + 0.05 * loss
        if time % 10 == 0 and writer:
            writer.add_scalar("loss", running_loss, time)
        if time % 1000 == 0:
            if time % 2500 == 0:
                acc = test_accuracy(net, accuracy_generator, flatten=flatten)
                acc_history.append((time, acc))
                if writer: writer.add_scalar("accuracy", acc, time)
                print(f"{time:>5d}\t{running_loss:.3f}\t{acc:.3f}")
            else:
                print(f"{time:>5d}\t{running_loss:.3f}")

        # stop training when we hit max_batches
        if time > max_batches:
            break

    # report final test accuracy
    acc = test_accuracy(net, accuracy_generator, max_batches=10000, flatten=flatten)
    if writer:
        writer.add_scalar("accuracy", acc, time)
        writer.close()
    print(f"Done training, final accuracy = {acc:.3f}")
    history = {'loss': loss_history, 'accuracy': acc_history}

    # return a bunch of statistics about the training run
    return {'loss': running_loss,               # final loss
            'accuracy': acc,                    # final test accuracy
            'history': history,                 # dictionary containing history of losses and test accuracies over time
            'weight_shapes': weight_shapes,     # list of shapes of trained arrays
            'weight_params': weight_params,     # list of names of trained arrays
            'batch_size': batch_size,           # batch size
            'batches': max_batches}             # number of batches to train for


def test_accuracy(net, generator, max_batches=5000, flatten=True):
    total = correct_total = 0
    for img, label, *rest in generator:
        if flatten:
            img = torch.flatten(img, start_dim=1)
        label_prime = net(img, *rest).argmax(1)
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
