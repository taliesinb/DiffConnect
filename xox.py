import torch

from torch import nn
from torch.nn import functional, Parameter


def yox(y, o, x):
    return torch.matmul(y, torch.matmul(o, torch.t(x)))


def bx(b, x):
    return torch.matmul(b, torch.t(x))


def make_ob(num_genes):
    o = torch.nn.Parameter(torch.randn(num_genes, num_genes))
    b = torch.nn.Parameter(torch.randn(num_genes))
    return o, b


class XOXHyperNetwork(object):
    def __init__(self, num_genes):
        self.num_genes = num_genes
        self.o_matrix = torch.nn.Parameter(torch.randn(num_genes, num_genes))
        self.b_matrix = torch.nn.Parameter(torch.randn(num_genes))

    def make_gene_matrix(self, num_units):
        return Parameter(torch.randn(num_units, self.num_genes))


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