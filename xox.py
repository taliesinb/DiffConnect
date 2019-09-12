import torch
from torch.nn import Parameter
import numpy as np
from hyper import HyperNetwork


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
        self.__post_init__()

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
