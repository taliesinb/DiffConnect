import torch
from torch import nn
from data import mnist, cifar

from hyper import RandomBasisHyperNetwork
from train import *
from param_xox import *
from utils import *

from indent import indent

train_params = {
    'iterator': mnist,
    'max_batches': 30000,
    'fields': ['weight_param_count', 'best_accuracy'],
    'test_interval': 1000,
    'max_parameters': 8000, # this skips models that produce more than this number of parameters
    'shuffle': True, # whether to visit the settings in a random order
    # 'subsample': 20, # this samples N elements from the full set of possibilities, for quick tests
    'runs': 20
}

train_params_nomax = {**train_params, 'max_parameters': None}

# these can be adjusted to train on e.g. CIFAR
io_params = {
    'ishape': [28, 28],
    'oshape': 10
}

###################################################################################################
## SLP AND MLP BASELINES                                                                         ##
###################################################################################################

baseline_params = {
    **io_params,
    'bias': TrueOrFalse
}

def TwoLayerLinear(ishape, oshape, bias=True):
    return MultilayerLinear([ishape, 800, oshape], bias=bias, nonlinearity='relu')

records = train_models([Linear, TwoLayerLinear], baseline_params, **train_params_nomax)
save_to_csv('csvs/baseline_mnist.csv', records)

###################################################################################################
## RANDOM BASIS                                                                                  ##
###################################################################################################

def RandomBasisOneLayer(ishape, oshape, ndims:int, bias=True):
    return RandomBasisHyperNetwork(Linear(ishape, oshape), ndims=ndims)

def RandomBasisTwoLayer(ishape, oshape, ndims:int, bias=True):
    return RandomBasisHyperNetwork(TwoLayerLinear(ishape, oshape), ndims=ndims)

rb_params = {
    **baseline_params,
    'ndims': LogIntegerRange(10, 1000, 20)
}

records = train_models([RandomBasisOneLayer, RandomBasisTwoLayer], rb_params, **train_params)
save_to_csv('csvs/scaling_rb_mnist.csv', records)

###################################################################################################
## XOX ONE LAYER                                                                                 ##
###################################################################################################

def OneLayerXOX(
        genes:int,
        ishape:list, oshape:int,
        is_input_learned:bool, is_input_gaussian:bool, is_readout_learned:bool,
        labels=0, label_nonlinearity='tanh',
        non_negative_weights=False, interaction_bias=False
    ):
    interaction = MaybeRelabeledProteinInteraction(genes, labels, nonlinearity=label_nonlinearity, include_bias=interaction_bias)
    input_expression = MaybeLearnedMaybeGaussianExpression[is_input_gaussian][is_input_learned](ishape)
    readout_expression = MaybeLearnedExpression[is_readout_learned](oshape)
    return XOXLinear(input_expression, readout_expression, interaction=interaction, non_negative=non_negative_weights)

xox_1_params = {
    **io_params,
    'genes': IntegerRange(30, 1),
    'is_input_learned': True,
    'is_input_gaussian': TrueOrFalse,
    'is_readout_learned': False,
    'interaction_bias': TrueOrFalse
}

records = train_models(OneLayerXOX, xox_1_params, **train_params_nomax)
save_to_csv('csvs/scaling_xox_1_mnist.csv', records)

###################################################################################################
## XOX TWO LAYER                                                                                 ##
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

    'genes': IntegerRange(30, 1),
    'is_input_learned': True,
    'is_hidden_learned': True,
    'is_readout_learned': False,
}

records = train_models(TwoLayerXOX, xox_2_params, **train_params)
save_to_csv('csvs/scaling_xox_2_mnist.csv', records)

