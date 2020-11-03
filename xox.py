import torch
from torch.nn import Parameter
import numpy as np
from hyper import HyperNetwork


spatial_dim = 28
grad1 = torch.linspace(-1, 1, spatial_dim)[:, None]
grad2 = torch.linspace(-1, 1, spatial_dim)[None, :]


class XOXHyperNetwork(HyperNetwork):

    def __init__(self, target_net, num_genes=8, skip_small=True, skip_vectors=True, symmetric=True, fix_o_matrix=False,
                 fix_gene_matrices=False, spatial_gene=False):
        self.num_genes = num_genes
        self.skip_small = skip_small
        self.skip_vectors = skip_vectors
        self.symmetric = symmetric
        self.spatial_gene = spatial_gene
        self.fix_gene_matrices = fix_gene_matrices
        super().__init__(target_net)
        self.o_matrix = torch.randn(num_genes, num_genes) * (1 / num_genes)
        self.b_matrix = torch.randn(num_genes) * np.sqrt(1 / num_genes)
        if not fix_o_matrix:
            self.o_matrix = Parameter(self.o_matrix)
            self.b_matrix = Parameter(self.b_matrix)
        self.__post_init__()

    def params(self):
        return self.

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
        if self.spatial_gene:
            self.fix_spatial(y)
            self.fix_spatial(x)
        res = torch.matmul(y, torch.matmul(self.o_matrix, torch.t(x)))
        return res

    def bx(self, x):
        if self.spatial_gene:
            self.fix_spatial(x)
        return torch.matmul(self.b_matrix, torch.t(x))

    def fix_spatial(self, p):
        n = p.shape[0]
        with torch.no_grad():
            if n == spatial_dim * spatial_dim:
                z = p.view(28, 28, -1)
                z[:, :, 0] = grad1
                z[:, :, 1] = grad2
            elif n == spatial_dim:
                z[:, 0] = grad1


if __name__ == '__main__':

    from cache import *
    from torch.nn import Linear
    from train import train
    from data import mnist_generator

    @cached
    def train_mnist_single_layer_xox(num_genes, steps=5000, spatial_gene=False):
        net = Linear(28 * 28, 10)
        hyper = XOXHyperNetwork(net, num_genes=num_genes, skip_small=False, skip_vectors=True, spatial_gene=spatial_gene)
        return train(net, mnist_generator, steps, title='hyper', hyper_net=hyper, log_dir=None)

    print(train_mnist_single_layer_xox(5, spatial_gene=True))
    print(train_mnist_single_layer_xox(3, spatial_gene=False))