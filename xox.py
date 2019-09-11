import torch

import torch.autograd

from torch.nn import Parameter, Module, Sequential, Linear, ReLU
from torch.nn.functional import mse_loss
from collections import OrderedDict
from utils import count_parameters

import numpy as np


def fix_name(name):
    return name.replace('.', '_')


class HyperNetwork(Module):

    def __init__(self, target_net):
        super().__init__()
        self.last_hyper_tensor = None
        target_params = OrderedDict(target_net.named_parameters())
        self.param_names = [fix_name(name) for name in target_params.keys()]
        self.lambdas = [self.make_hyper_lambda(fix_name(name), param) for name, param in target_params.items()]
        self.outputs = None
        self.__dict__['target_params'] = list(target_params.values())
        print(f"Hypernetwork: {count_parameters(target_net)} -> {count_parameters(self)}")
        # ^ use of __dict__ stops us from owning the params

    def forward(self, set_arrays=True):
        self.outputs = [fn() for fn in self.lambdas]
        if set_arrays:
            with torch.no_grad():
                return [dst.copy_(src) for src, dst in zip(self.outputs, self.target_params)]
        return self.outputs

    def backward(self):
        grad_tensors = [p.grad for p in self.target_params]
        torch.autograd.backward(self.outputs, grad_tensors)

    def discrepancy_loss(self):
        loss = sum(mse_loss(src, dst.detach(), reduction='sum')
                   for src, dst in zip(self.outputs, self.target_params))
        return loss

    def make_hyper_lambda(self, name, param):
        raise NotImplementedError()

    def make_hyper_tensor(self, name, shape, var=1.0, set_last=True, fix=False):
        param = torch.randn(shape) * np.sqrt(var)
        param_name = 'hyper_' + name
        if set_last: self.last_hyper_tensor = param_name
        if not fix:
            param = Parameter(param)
            self.register_parameter(param_name, param)
        return param

    def get_last_hyper_tensor(self):
        return self._parameters[self.last_hyper_tensor]


class DummyHyperNetwork(HyperNetwork):

    def make_hyper_lambda(self, name, param):
        param = self.make_hyper_tensor(name, param.shape)
        return lambda: param

    def absorb(self, step_size=0.1):
        with torch.no_grad():
            for dst, name in zip(self.target_params, self.param_names):
                src = self._parameters['hyper_' + name]
                src.copy_(src * (1-step_size) + dst * step_size)


class HaHypernetwork(HyperNetwork):

    def make_hyper_lambda(self, name, param):
        param = self.make_hyper_tensor(name, param.shape)
        return lambda: param

    def absorb(self, step_size=0.1):
        with torch.no_grad():
            for dst, name in zip(self.target_params, self.param_names):
                src = self._parameters['hyper_' + name]
                src.copy_(src * (1-step_size) + dst * step_size)


class XOXHyperNetwork(HyperNetwork):

    def __init__(self, target_net, num_genes=8, skip_small=True, skip_vectors=True, symmetric=True, fix_o_matrix=False,
                 fix_gene_matrices=False):
        self.num_genes = num_genes
        self.skip_small = skip_small
        self.skip_vectors = skip_vectors
        self.symmetric = symmetric
        self.fix_gene_matrices = fix_gene_matrices
        super().__init__(target_net)
        self.o_matrix = torch.randn(num_genes, num_genes) * (1 / num_genes)
        self.b_matrix = torch.randn(num_genes) * np.sqrt(1 / num_genes)
        if not fix_o_matrix:
            self.o_matrix = Parameter(self.o_matrix)
            self.b_matrix = Parameter(self.b_matrix)

    def should_share_x(self, name, num_x):
        return (
            self.symmetric and
            self.last_hyper_tensor and
            self.last_hyper_tensor.endswith(('_weight_y', '_bias_y')) and
            name.endswith('_weight') and
            self.get_last_hyper_tensor().shape[0] == num_x
        )

    def should_share_y(self, name):
        return (
            not self.skip_vectors and
            self.last_hyper_tensor == 'hyper_' + name.replace('bias', 'weight') + '_y'
        )

    def make_hyper_lambda(self, name, param):
        shape = param.shape
        size = np.prod(shape)
        is_small = size <= sum(shape) * self.num_genes
        # if it is a matrix
        if len(shape) == 2 and not (self.skip_small and is_small):
            num_y, num_x = shape
            var = 1 / np.sqrt(3 * num_x)
            if self.should_share_x(name, num_x):
                x_genes = self.get_last_hyper_tensor()
            else:
                x_genes = self.make_hyper_tensor(name + '_x', (num_x, self.num_genes), var=var, fix=self.fix_gene_matrices)
            y_genes = self.make_hyper_tensor(name + '_y', (num_y, self.num_genes), var=var)
            return lambda: self.yox(y_genes, x_genes)
        # if it is a vector
        if len(shape) == 1 and self.should_share_y(name):
            y_genes = self.get_last_hyper_tensor()
            return lambda: self.bx(y_genes)
        # just model the parameter directly
        param = self.make_hyper_tensor(name, param.shape, set_last=False, var=(param.var().item()))
        return lambda: param

    def yox(self, y, x):
        return torch.matmul(y, torch.matmul(self.o_matrix, torch.t(x)))

    def bx(self, x):
        return torch.matmul(self.b_matrix, torch.t(x))
