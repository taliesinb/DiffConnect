import torch, torch.nn.functional
import numpy as np
import hyper
import utils
import indent

#@indent.indenting
def train(net, iterator_factory, *,
        max_batches=2000,
        loss_fn=utils.cross_entropy_loss,
        optimizer='Adam', lr=0.01,
        log_dir='runs', title=None, 
        batch_size=64, flatten=True):

    # if a hypernetwork was provided, we don't directly train via the network parameters,
    # we train via the hypernetwork parameters (which produce the network parameters)
    if isinstance(net, hyper.HyperNetwork):
        hyper_net, net = net, net.target_net
        params = hyper_net.parameters()
    else:
        hyper_net = None
        params = net.parameters()

    params = list(params)
    weight_shapes = [tuple(p.shape) for p in params]
    weight_params = sum(np.prod(shape) for shape in weight_shapes)
    title_str = f'Training "{title}"' if title else 'Training'
    shape_str = ' '.join('×'.join(map(str, x)) for x in weight_shapes)
    print(f"{title_str} (arrays: {len(params)}, params: {weight_params}, shapes: {shape_str})")

    writer = utils.make_log_writer(log_dir, title)
    optimizer = utils.opt_mapping[optimizer](params, lr=lr)

    training_iterator, test_iterator = iterator_factory(batch_size=batch_size, flatten=flatten)
    
    running_loss = None
    loss_history = []
    acc_history = []

    # main training loop
    for batch_num in range(max_batches):
        optimizer.zero_grad()
        if hyper_net:  # use hypernetwork (if any) to derive the parameters for our network
            net.zero_grad()
            hyper_net.zero_grad()
            hyper_net.forward()
            hyper_net.push_weights()
        loss = loss_fn(net, next(training_iterator))
        loss.backward()
        if hyper_net: # backprop from ordinary grads to hypergrads
                hyper_net.backward() 
        optimizer.step()

        # report losses, etc.
        loss = loss.item()
        loss_history.append(loss)
        running_loss = (0.95 * running_loss + 0.05 * loss) if running_loss else loss
        if batch_num % 10 == 0 and writer:
            writer.add_scalar("loss", running_loss, batch_num)
        if batch_num % 1000 == 0:
            if batch_num % 2500 == 0:
                acc = utils.test_accuracy(net, test_iterator)
                acc_history.append((batch_num, acc))
                if writer: writer.add_scalar("accuracy", acc, batch_num)
                print(f"{batch_num:>5d}\t{running_loss:.3f}\t{acc:.3f}")
            else:
                print(f"{batch_num:>5d}\t{running_loss:.3f}")

    # report final test accuracy
    acc = utils.test_accuracy(net, test_iterator, max_items=10000)
    if writer:
        writer.add_scalar("accuracy", acc, batch_num)
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
