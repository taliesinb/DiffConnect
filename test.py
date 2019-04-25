from importlib import reload

import numpy as np
import torch
from torch import nn
import diff_connect as dc


mnist = dc.mnist_data()
mnist_test = dc.mnist_data(is_train=False)

train_args = {
    'regression':False, 'data': mnist_test,
    'epochs': 5, 'sample_every':50, 'optimizer':'SGD',
    'print_loss': True, 'lr': 0.0001
}

x = np.random.randn(1, 28, 28)

mlp_net = nn.Sequential(dc.flatten(), nn.Linear(28 ** 2, 100), nn.ReLU(), nn.Linear(100, 10))

reload(dc.linear_dc);
reload(dc);

dc_net = nn.Sequential(dc.flatten(), dc.LinearDC(28 ** 2, 10, 11, 5))
dc_net(dc.toT(x))

result = dc.train_net(dc_net, **train_args)

plt.title("loss curves")
plt.ylim(0.07, 2)
