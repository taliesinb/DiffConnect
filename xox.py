import torch

from torch import nn
from torch.nn import functional, Parameter
import numpy as np


def yox(y, o, x):
    return torch.matmul(y, torch.matmul(o, torch.t(x)))


def bx(b, x):
    return torch.matmul(b, torch.t(x))


def make_ob(num_genes):
    o = torch.nn.Parameter(torch.randn(num_genes, num_genes))
    b = torch.nn.Parameter(torch.randn(num_genes))
    return o, b


class XOXHyperNetwork(nn.Module):
    def __init__(self, num_genes, sharing=False, freeze=False):
        super().__init__()
        self.num_genes = num_genes
        self.sharing = False
        self.o_matrix = torch.randn(num_genes, num_genes)
        self.b_matrix = torch.randn(num_genes)
        self.prev = None
        self.sharing = sharing
        if not freeze:
            self.o_matrix = torch.nn.Parameter(self.o_matrix)
            self.b_matrix = torch.nn.Parameter(self.b_matrix)

    def set_prev(self, prev):
        if self.sharing:
            self.prev = prev

    def linear(self, num_inputs, num_outputs):
        layer = XOXLinear(num_inputs, num_outputs, hyper=self, prev=self.prev)
        self.set_prev(layer)
        return layer

    def linear_2d(self, num_inputs, num_outputs, spatial_x=True, spatial_y=True):
        layer = XOXLinear2D(num_inputs, num_outputs, spatial_x=spatial_x, spatial_y=spatial_y, hyper=self, prev=self.prev)
        self.set_prev(layer)
        return layer

    def make_gene_matrix(self, num_units):
        return Parameter(torch.randn(num_units, self.num_genes))


'''
            if rt * rt == num_units:
                rt = np.round(np.sqrt(num_units))
                pattern = [(i / rt, j / rt) for i in range(-rt+1, rt, 2) for j in range(-rt+1, rt)]
                matrix[:, 0] = pattern[:, 0]
                matrix[:, 1] = pattern[:, 1]
'''


class XOXLinear(nn.Module):
    def __init__(self, num_inputs, num_outputs, *, hyper=None, prev=None):
        super().__init__()
        if prev:
            self.hyper = prev.hyper
            self.x_genes = prev.y_genes
            self.y_genes = prev.hyper.make_gene_matrix(num_outputs)
        else:
            assert hyper
            self.hyper = hyper
            self.x_genes = hyper.make_gene_matrix(num_inputs)
            self.y_genes = hyper.make_gene_matrix(num_outputs)

    def forward(self, x):
        weight = yox(self.y_genes, self.hyper.o_matrix, self.x_genes)
        bias = bx(self.hyper.b_matrix, self.y_genes)
        return functional.linear(x, weight, bias)


def make_2d_pattern(n):
    rng = np.arange(-1, 1 + 1e-5, 2/(n-1))
    return torch.tensor([[i, j] for i in rng for j in rng])


def make_1d_pattern(n):
    return torch.arange(-1, 1 + 1e-5, 2/(n-1))


def int_sqrt(n):
    sqrt = np.sqrt(n)
    sqrt_r = round(sqrt)
    assert sqrt == sqrt_r, f"cannot make a spatial substrate for non-square number of neurons {n}"
    return sqrt_r


class XOXLinear2D(XOXLinear):
    def __init__(self, num_inputs, num_outputs, *, spatial_x=True, spatial_y=True, hyper=None, prev=None):
        super().__init__(num_inputs, num_outputs, hyper=hyper, prev=prev)
        self.x_mask = make_2d_pattern(int_sqrt(num_inputs)) if spatial_x else None
        self.y_mask = make_2d_pattern(int_sqrt(num_outputs)) if spatial_y else None

    def forward(self, x):
        with torch.no_grad():
            if self.x_mask is not None:
                self.x_genes[:, :2] = self.x_mask
            if self.y_mask is not None:
                self.y_genes[:, :2] = self.y_mask
        return super().forward(x)