import torch
from torch import nn
from data import mnist, cifar,fmnist,kmnist

from hyper import RandomBasisHyperNetwork
from train import *
from param_xox import *
from utils import *
from colorama import Fore, Back, Style


from indent import indent

train_params = {
    'max_batches': 30000,
    'test_interval': 1000,
    'max_parameters': 8000 # this skips models that produce more than this number of parameters
}

train_params_nomax = {**train_params, 'max_parameters': None}
io_params = {
    'ishape': [28, 28],
    'oshape': 10
}

# these can be adjusted to train on e.g. CIFAR
ishape = [28, 28]
oshape = 10


###################################################################################################
## MLP BASELINE                                                                                  ##
###################################################################################################
# mlp = MultilayerLinear([[28,28], 100, 10], bias=False, nonlinearity='relu')
# res = train(mlp,mnist, **train_params_nomax)
# utils.print_image(torch.reshape(mlp[1].weight.detach(),(100,28,28)))
# utils.print_image(torch.reshape(mlp[3].weight.detach(),(10,10,10)))


###################################################################################################
## RANDOM BASIS                                                                                  ##
###################################################################################################
# def TwoLayerLinear(ishape, oshape, bias=False):
#     return MultilayerLinear([ishape, 100, oshape], bias=bias, nonlinearity='relu')

# rb_mlp = RandomBasisHyperNetwork(TwoLayerLinear(ishape, oshape), ndims=800)
# res = train(rb_mlp,mnist, **train_params_nomax)
# utils.print_image(torch.reshape(rb_mlp.target_net[1].weight.detach(),(100,28,28)))
# utils.print_image(torch.reshape(rb_mlp.target_net[3].weight.detach(),(10,10,10)))

###################################################################################################
## XOX TWO LAYER                                                                                 ##
###################################################################################################
# def TwoLayerNonSpatialXOX(
#         genes:int,
#         ishape:list,hshape:list, oshape:int,
#         is_input_learned:bool,is_hidden_learned:bool, is_input_gaussian:bool, is_readout_learned:bool,
#         is_expression_shared:bool, is_interaction_shared:bool,
#         labels=0, label_nonlinearity='tanh',
#         non_negative_weights=False, interaction_bias=False
#     ):
#     interaction_1, interaction_2 = maybe_shared(is_interaction_shared, lambda: ProteinInteraction(genes))
#     input_expression = MaybeLearnedMaybeGaussianExpression[is_input_gaussian][is_input_learned](ishape)
#     hidden_expression_1, hidden_expression_2 = maybe_shared(is_expression_shared, lambda: MaybeLearnedMaybeGaussianExpression[is_input_gaussian][is_hidden_learned](hshape))
#     readout_expression = MaybeLearnedExpression[is_readout_learned](oshape)
#     return XOXSequential(
#             XOXLinear(input_expression, hidden_expression_1, interaction=interaction_1),
#             XOXLinear(hidden_expression_2, readout_expression, interaction=interaction_2)
#         )

# xox_2_params = {
#     **io_params,
#     'hshape': [10, 10],
#     'is_expression_shared': True,
#     'is_interaction_shared': True,
#     'is_input_gaussian': False,
#     'genes': 25,
#     'is_input_learned': True,
#     'is_hidden_learned': True,
#     'is_readout_learned': False,
# }
# xox_nonspatial = TwoLayerNonSpatialXOX(**xox_2_params)
# res = train(xox_nonspatial,mnist, **train_params_nomax)
# utils.print_image(torch.reshape(xox_nonspatial[0].calculate_weights().detach(),(100,28,28)))
# utils.print_image(torch.reshape(xox_nonspatial[1].calculate_weights().detach(),(10,10,10)))

###################################################################################################
## XOX TWO LAYER   Spatial                                                                       ##
###################################################################################################
def TwoLayerXOX(
        genes:int,
        ishape:list, hshape:list, oshape:int,
        is_input_learned:bool, is_hidden_learned:bool, is_readout_learned:bool,
        is_expression_shared:bool, is_interaction_shared:bool
    ):
    if is_expression_shared and not is_hidden_learned:
        print("warning: is_expression_shared=True has no effect when is_hidden_learned=False")
    interaction_1, interaction_2 = maybe_shared(is_interaction_shared, lambda: ProteinInteraction(genes))
    input_expression = MaybeLearnedGaussianExpression[is_input_learned](ishape)
    hidden_expression_1, hidden_expression_2 = maybe_shared(is_expression_shared, lambda: MaybeLearnedGaussianExpression[is_hidden_learned](hshape))
    readout_expression = MaybeLearnedExpression[is_readout_learned](oshape)
    return XOXSequential(
        XOXLinear(input_expression, hidden_expression_1, interaction=interaction_1),
        XOXLinear(hidden_expression_2, readout_expression, interaction=interaction_2)
    )

xox_2_params = {
    **io_params,
    'hshape': [28, 28],
    'is_expression_shared': True,
    'is_interaction_shared': True,
    'genes': 10,
    'is_input_learned': True,
    'is_hidden_learned': True,
    'is_readout_learned': True
}

lr_frozen = 0.001

###################################################################################################
## XOX TWO LAYER  Transfer                                                                       ##
###################################################################################################
print(Fore.YELLOW, f"Performance on FMNIST with all but last layer frozen.", Fore.RESET)

xox_spatial = TwoLayerXOX(**xox_2_params)

for key,value in xox_spatial.named_parameters():
    if key != '1.output:gene_matrix':
        value.requires_grad = False

res = train(xox_spatial,fmnist, lr=lr_frozen, **train_params_nomax)

print(Fore.YELLOW, f"Performance on MNIST.", Fore.RESET)
xox_spatial = TwoLayerXOX(**xox_2_params)

res = train(xox_spatial,mnist, **train_params_nomax)
print(Fore.YELLOW, f"Performance on FMNIST after MNIST.", Fore.RESET)

for key,value in xox_spatial.named_parameters():
    if key != '1.output:gene_matrix':
        value.requires_grad = False

res = train(xox_spatial,fmnist, lr=lr_frozen, **train_params_nomax)

print(Fore.YELLOW, f"Performance on KMNIST.", Fore.RESET)
xox_spatial = TwoLayerXOX(**xox_2_params)

res = train(xox_spatial,kmnist, **train_params_nomax)

print(Fore.YELLOW, f"Performance on FMNIST after KMNIST.", Fore.RESET)

for key,value in xox_spatial.named_parameters():
    if key != '1.output:gene_matrix':
        value.requires_grad = False

res = train(xox_spatial,fmnist, lr=lr_frozen, **train_params_nomax)


print(Fore.YELLOW, f"Performance on MNIST with all but last layer frozen.", Fore.RESET)

xox_spatial = TwoLayerXOX(**xox_2_params)

for key,value in xox_spatial.named_parameters():
    if key != '1.output:gene_matrix':
        value.requires_grad = False

res = train(xox_spatial,mnist, lr=lr_frozen, **train_params_nomax)

print(Fore.YELLOW, f"Performance on FMNIST.", Fore.RESET)

xox_spatial = TwoLayerXOX(**xox_2_params)

res = train(xox_spatial,fmnist, **train_params_nomax)

for key,value in xox_spatial.named_parameters():
    if key != '1.output:gene_matrix':
        value.requires_grad = False

print(Fore.YELLOW, f"Performance on MNIST after FMNIST.", Fore.RESET)

res = train(xox_spatial,mnist, lr=lr_frozen, **train_params_nomax)

print(Fore.YELLOW, f"Performance on KMNIST.", Fore.RESET)

xox_spatial = TwoLayerXOX(**xox_2_params)

res = train(xox_spatial,kmnist, **train_params_nomax)

for key,value in xox_spatial.named_parameters():
    if key != '1.output:gene_matrix':
        value.requires_grad = False

print(Fore.YELLOW, f"Performance on MNIST after KMNIST.", Fore.RESET)
res = train(xox_spatial,mnist, lr=lr_frozen, **train_params_nomax)


print(Fore.YELLOW, f"Performance on KMNIST with all but last layer frozen.", Fore.RESET)

xox_spatial = TwoLayerXOX(**xox_2_params)

for key,value in xox_spatial.named_parameters():
    if key != '1.output:gene_matrix':
        value.requires_grad = False

res = train(xox_spatial,kmnist, lr=lr_frozen, **train_params_nomax)

print(Fore.YELLOW, f"Performance on FMNIST.", Fore.RESET)
xox_spatial = TwoLayerXOX(**xox_2_params)

res = train(xox_spatial,fmnist, **train_params_nomax)

for key,value in xox_spatial.named_parameters():
    if key != '1.output:gene_matrix':
        value.requires_grad = False

print(Fore.YELLOW, f"Performance on KMNIST after FMNIST.", Fore.RESET)

res = train(xox_spatial,kmnist, lr=lr_frozen, **train_params_nomax)

print(Fore.YELLOW, f"Performance on MNIST.", Fore.RESET)
xox_spatial = TwoLayerXOX(**xox_2_params)

res = train(xox_spatial,mnist, **train_params_nomax)

for key,value in xox_spatial.named_parameters():
    if key != '1.output:gene_matrix':
        value.requires_grad = False

print(Fore.YELLOW, f"Performance on KMNIST after MNIST.", Fore.RESET)
res = train(xox_spatial,kmnist, lr=lr_frozen, **train_params_nomax)