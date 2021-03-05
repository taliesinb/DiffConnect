import torch
from torch import nn
from data import mnist, cifar

from hyper import RandomBasisHyperNetwork
from train import *
from param_xox import *

from indent import indent
from utils import product, save_to_csv, count_parameters

def OneLayerXOX(
        genes:int,
        ishape:list, oshape:int,
        is_input_learned:bool, is_input_gaussian:bool, is_readout_learned:bool
    ):
    interaction = ProteinInteraction(genes)
    input_expression = MaybeLearnedMaybeGaussianExpression[is_input_gaussian][is_input_learned](ishape)
    readout_expression = MaybeLearnedExpression[is_readout_learned](oshape)
    return XOXLinear(input_expression, readout_expression, interaction=interaction)

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

train_params = {
    'iterator': mnist,
    'max_batches': 10000,
    'fields': ['weight_param_count', 'best_accuracy'],
    'max_parameters': 1000, # this skips models that produce more than this number of parameters
    'shuffle': False,
    'subsample': 5, # this samples N elements from the full set of possibilities, for quick tests
    'runs': 1
}

one_layer_params = {
    'ishape': [28, 28],
    'oshape': 10,

    'genes': IntegerRange(30, 1),
    'is_input_learned': TrueOrFalse,
    'is_input_gaussian': TrueOrFalse,
    'is_readout_learned': TrueOrFalse,
}

records = train_models(OneLayerXOX, one_layer_params, **train_params)
save_to_csv('scaling_one_layer_mnist.csv', records)

two_layer_params = {
    'ishape': [28, 28],
    'hshape': [10, 10],
    'oshape': 10,

    'is_expression_shared': True,
    'is_interaction_shared': True,

    'genes': IntegerRange(30, 1),
    'is_input_learned': TrueOrFalse,
    'is_hidden_learned': TrueOrFalse,
    'is_readout_learned': TrueOrFalse,
}

records = train_models(TwoLayerXOX, two_layer_params, **train_params)
save_to_csv('scaling_two_layers_mnist.csv', records)

