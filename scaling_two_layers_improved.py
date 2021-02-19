import torch
from torch import nn
from data import mnist, cifar

from hyper import RandomBasisHyperNetwork
from train import train, cached_train, train_models
from param_xox import *

from indent import indent
from utils import product, save_to_csv

ishape = (28, 28)
oshape = 10

def maybe_shared(is_shared:bool, fn):
    if is_shared:
        expr1 = expr2 = fn()
    else:
        expr1 = fn()
        expr2 = fn()
    return expr1, expr2

MaybeLearnedGaussianExpression = {True: LearnedGaussianExpression, False: RandomGaussianExpression}
MaybeLearnedExpression = {True: LearnedExpression, False: RandomExpression}

def TwoLayerXOX(genes:int, hshape:list, is_first_learned:bool, is_hidden_learned:bool, is_readout_learned:bool, is_expression_shared:bool, is_interaction_shared:bool):
    if is_expression_shared and not is_hidden_learned: 
        print("warning: is_expression_shared=True has no effect when is_hidden_learned=False")
    interaction_1, interaction_2 = maybe_shared(is_interaction_shared, lambda: ProteinInteraction(genes))
    first_expression = MaybeLearnedGaussianExpression[is_first_learned](ishape)
    hidden_expression_1, hidden_expression_2 = maybe_shared(is_expression_shared, lambda: MaybeLearnedGaussianExpression[is_hidden_learned](hshape))
    readout_expression = MaybeLearnedExpression[is_readout_learned](oshape)
    return XOXSequential(
        XOXLinear(first_expression, hidden_expression_1, interaction=interaction_1),
        XOXLinear(hidden_expression_2, readout_expression, interaction=interaction_2)
    )

'''
# this is to test that all the model parameters are as expected for all combinations of settings (visually)
bools = [False, True]
for is_first_learned in bools:
    for is_hidden_learned in bools:
        for is_readout_learned in bools:
            for is_expression_shared in bools:
                for is_interaction_shared in bools:
                    print(f"{is_first_learned=} {is_hidden_learned=} {is_readout_learned=} {is_expression_shared=} {is_interaction_shared=}")
                    model = TwoLayerXOX(5, [10,10], is_first_learned, is_hidden_learned, is_readout_learned, is_expression_shared, is_interaction_shared)
                    utils.print_model_parameters(model)

exit(0)
'''

bools = [False, True]
params = {
    'genes': range(1,20),
    'hshape': [[10, 10]],
    'is_first_learned': bools, 'is_hidden_learned': bools, 'is_readout_learned': bools,
    'is_expression_shared': [True], 'is_interaction_shared': [True]
}

num_steps = 10000
runs = 10
records = train_models([TwoLayerXOX], params, mnist, max_batches=num_steps, fields=['weight_param_count', 'best_accuracy'], runs=runs)
save_to_csv('scaling_two_layers_improved.csv', records)
