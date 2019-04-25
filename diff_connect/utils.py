import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import math

class flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)

def toT(x):
    return torch.tensor(x, dtype=torch.float32, requires_grad=False)

def fromT(x):
    return x.detach().numpy()

def to_trigger(spec):
    if isinstance(spec, tuple):
        interval = spec[0]
        func = spec[1]
    else:
        interval = 'batch'
        func = spec

    if isinstance(interval, int):
        trigger = lambda b, e: b % interval == 0
    elif isinstance(interval, list):
        trigger = lambda b, e: b in interval
    elif interval == 'batch':
        trigger = lambda b, e: True
    elif interval == 'epoch':
        trigger = lambda b, e: e == round(e)
    else:
        raise Exception('bad callback trigger')

    return lambda n, b, e: (func({'net':n, 'batch':b, 'epoch':e}) if trigger(b, e) else None)

def train_net(net, data, epochs=50, sample_every=5, print_loss=False, live_plot=False, regression=True, updates=1, optimizer='SGD', lr=0.01, callbacks=[]):
    criterion = nn.MSELoss() if regression else nn.CrossEntropyLoss()

    if isinstance(optimizer, str):
        optclass = getattr(optim, optimizer)
        optfactory = lambda params: optclass(params, lr=lr)
    elif isinstance(optimizer, dict):
        optimizer = optimizer.copy()
        opttype = optimizer.pop("type")
        optclass = getattr(optim, opttype)
        optfactory = lambda params: optclass(params, lr=lr, **optimizer)
    else:
        raise Exception('bad optimizer')

    optimizer = optfactory(net.parameters())
    losses = []

    if live_plot:
        plt.ion()
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(1,1,1)

    tick = 1
    running_loss = 0.0

    triggers = [to_trigger(s) for s in callbacks]

    n_epochs = math.ceil(epochs)
    batches_per_epoch = len(data)

    batch_index = 0
    max_batch_index = math.ceil(batches_per_epoch * epochs)

    for epoch in range(n_epochs):
        for batch in data:
            batch_index += 1
            if batch_index == max_batch_index:
                break

            for trigger in triggers:
                trigger(net, batch_index, epoch)

            inputs, targets = batch

            batch_loss = None
            for u in range(updates):
                net.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                if batch_loss == None:
                    batch_loss = loss.item()
                loss.backward()

            optimizer.step()

            running_loss += batch_loss
            tick += 1
            if tick % sample_every == 1:
                running_loss /= sample_every
                losses.append(running_loss)
                if print_loss:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_index, running_loss))
                if live_plot:
                    ax.clear()
                    ax.semilogy(losses)
                    fig.canvas.draw()
                running_loss = 0.0

    if live_plot:
        plt.ioff()
        plt.cla()
        plt.close()

    return {'loss': losses, 'net': net}

def test_categorical_accuracy(net, data_loader):
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = net(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(data_loader.dataset)

def mnist_data(batch_size=64, is_train=True):
	ts = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	data = datasets.MNIST('data', train=is_train, download=True, transform=ts)
	return DataLoader(data, batch_size=batch_size, shuffle=True)
