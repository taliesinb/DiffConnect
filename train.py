import torch

from torch.nn import functional, Parameter

import tensorboardX


run_count = 0


def train(net, generator_factory, max_batches, *, lr=0.01, title=None, batch_size=64, log_weights=False):
    global run_count
    params = list(net.parameters())
    num_params = sum(p.numel() for p in params)
    print("Param sizes:")
    for p in params:
        print(p.shape)
    print(f"Number of parameters = {num_params}")
    if title is None:
        run_count += 1
        title = f"{run_count:03d}"
    writer = tensorboardX.SummaryWriter(f"runs/{title:s}", flush_secs=2)
    opt = torch.optim.Adam(params, lr=lr)
    time = 0
    accuracy_generator = generator_factory(batch_size, is_training=True)
    training_generator = generator_factory(batch_size, is_training=False)
    running_loss = None
    for img, label in training_generator:
        time += 1
        opt.zero_grad()
        img = torch.flatten(img, start_dim=1)
        label_prime = net(img)
        loss = functional.cross_entropy(label_prime, label)
        loss.backward()
        opt.step()
        loss = loss.item()
        if running_loss is None:
            running_loss = loss
        else:
            running_loss = 0.95 * running_loss + 0.05 * loss
        if time % 10 == 0:
            writer.add_scalar("loss", running_loss, time)
        if log_weights and (time == 1 or time % 2000 == 0):
            writer.add_histogram("o_matrix", net[0].hyper.o_matrix.flatten(start_dim=0), time)
            writer.add_histogram("b_matrix", net[0].hyper.b_matrix.flatten(start_dim=0), time)
            writer.add_histogram("0/x_genes", net[0].x_genes, time)
            writer.add_histogram("0/y_genes", net[0].y_genes, time)
            writer.add_histogram("2/x_genes", net[2].x_genes, time)
            writer.add_histogram("2/y_genes", net[2].y_genes, time)
        if time % 500 == 0:
            if time % 2500 == 0:
                acc = test_accuracy(net, accuracy_generator)
                writer.add_scalar("accuracy", acc, time)
                print(f"{time:>5d}\t{loss:.3f}\t{acc:.3f}")
            else:
                print(f"{time:>5d}\t{loss:.3f}")
        if time > max_batches:
            break
    acc = test_accuracy(net, accuracy_generator, max_batches=10000)
    writer.add_scalar("accuracy", acc, time)
    writer.close()
    print(f"final accuracy = {acc:.3f}")
    return {'loss': running_loss, 'accuracy': acc}


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
