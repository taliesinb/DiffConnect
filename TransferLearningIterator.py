import torch
from torch import nn
from data import mnist, cifar,fmnist,kmnist
import pandas as pd

from hyper import RandomBasisHyperNetwork
from train import *
from param_xox import *
from utils import *
from colorama import Fore, Back, Style
import copy



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


lr_frozen = 0.001

###################################################################################################
## XOX TWO Spatial LAYER  Transfer    -- updates x_o only                                        ##
###################################################################################################
Accuracies = []
ModelNames = []
Genes = []
NumReps = []
for reps in range(5):
    for g in reversed(range(1,30)):
        xox_2_params = {
            **io_params,
            'hshape': [28, 28],
            'is_expression_shared': True,
            'is_interaction_shared': True,
            'genes': g,
            'is_input_learned': True,
            'is_hidden_learned': True,
            'is_readout_learned': True
        }
        print(Fore.YELLOW, f"Performance on FMNIST with all but last layer frozen.", Fore.RESET)

        xox_spatial = TwoLayerXOX(**xox_2_params)
        for key,value in xox_spatial.named_parameters():
            if key != '1.output:gene_matrix':
                value.requires_grad = False
        res = train(xox_spatial,fmnist, lr=lr_frozen, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        Genes.append(g)
        NumReps.append(reps)
        ModelNames.append('FMNIST_Frozen')

        print(Fore.YELLOW, f"Performance on MNIST.", Fore.RESET)

        xox_spatial = TwoLayerXOX(**xox_2_params)
        res = train(xox_spatial,mnist, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        Genes.append(g)
        NumReps.append(reps)
        ModelNames.append('MNIST')

        print(Fore.YELLOW, f"Performance on FMNIST after MNIST.", Fore.RESET)

        for key,value in xox_spatial.named_parameters():
            if key != '1.output:gene_matrix':
                value.requires_grad = False
        res = train(xox_spatial,fmnist, lr=lr_frozen, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        Genes.append(g)
        NumReps.append(reps)
        ModelNames.append('FMNIST_After_MNIST')

        print(Fore.YELLOW, f"Performance on KMNIST.", Fore.RESET)

        xox_spatial = TwoLayerXOX(**xox_2_params)
        res = train(xox_spatial,kmnist, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        Genes.append(g)
        NumReps.append(reps)
        ModelNames.append('KMNIST')

        print(Fore.YELLOW, f"Performance on FMNIST after KMNIST.", Fore.RESET)

        for key,value in xox_spatial.named_parameters():
            if key != '1.output:gene_matrix':
                value.requires_grad = False
        res = train(xox_spatial,fmnist, lr=lr_frozen, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        Genes.append(g)
        NumReps.append(reps)
        ModelNames.append('FMNIST_After_KMNIST')

        print(Fore.YELLOW, f"Performance on MNIST with all but last layer frozen.", Fore.RESET)

        xox_spatial = TwoLayerXOX(**xox_2_params)
        for key,value in xox_spatial.named_parameters():
            if key != '1.output:gene_matrix':
                value.requires_grad = False
        res = train(xox_spatial,mnist, lr=lr_frozen, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        Genes.append(g)
        NumReps.append(reps)
        ModelNames.append('MNIST_Frozen')

        print(Fore.YELLOW, f"Performance on FMNIST.", Fore.RESET)

        xox_spatial = TwoLayerXOX(**xox_2_params)
        res = train(xox_spatial,fmnist, **train_params_nomax)
        for key,value in xox_spatial.named_parameters():
            if key != '1.output:gene_matrix':
                value.requires_grad = False

        Accuracies.append(res.get('best_accuracy'))
        Genes.append(g)
        NumReps.append(reps)
        ModelNames.append('FMNIST')

        print(Fore.YELLOW, f"Performance on MNIST after FMNIST.", Fore.RESET)

        res = train(xox_spatial,mnist, lr=lr_frozen, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        Genes.append(g)
        NumReps.append(reps)
        ModelNames.append('MNIST_After_FMNIST')

        print(Fore.YELLOW, f"Performance on KMNIST.", Fore.RESET)

        xox_spatial = TwoLayerXOX(**xox_2_params)
        res = train(xox_spatial,kmnist, **train_params_nomax)
        for key,value in xox_spatial.named_parameters():
            if key != '1.output:gene_matrix':
                value.requires_grad = False

        Accuracies.append(res.get('best_accuracy'))
        Genes.append(g)
        NumReps.append(reps)
        ModelNames.append('KMNIST')

        print(Fore.YELLOW, f"Performance on MNIST after KMNIST.", Fore.RESET)

        res = train(xox_spatial,mnist, lr=lr_frozen, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        Genes.append(g)
        NumReps.append(reps)
        ModelNames.append('MNIST_After_KMNIST')

        print(Fore.YELLOW, f"Performance on KMNIST with all but last layer frozen.", Fore.RESET)

        xox_spatial = TwoLayerXOX(**xox_2_params)
        for key,value in xox_spatial.named_parameters():
            if key != '1.output:gene_matrix':
                value.requires_grad = False
        res = train(xox_spatial,kmnist, lr=lr_frozen, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        Genes.append(g)
        NumReps.append(reps)
        ModelNames.append('KMNIST_Frozen')

        print(Fore.YELLOW, f"Performance on FMNIST.", Fore.RESET)

        xox_spatial = TwoLayerXOX(**xox_2_params)
        res = train(xox_spatial,fmnist, **train_params_nomax)
        for key,value in xox_spatial.named_parameters():
            if key != '1.output:gene_matrix':
                value.requires_grad = False

        Accuracies.append(res.get('best_accuracy'))
        Genes.append(g)
        NumReps.append(reps)
        ModelNames.append('FMNIST')

        print(Fore.YELLOW, f"Performance on KMNIST after FMNIST.", Fore.RESET)

        res = train(xox_spatial,kmnist, lr=lr_frozen, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        Genes.append(g)
        NumReps.append(reps)
        ModelNames.append('KMNIST_After_FMNIST')

        print(Fore.YELLOW, f"Performance on MNIST.", Fore.RESET)

        xox_spatial = TwoLayerXOX(**xox_2_params)
        res = train(xox_spatial,mnist, **train_params_nomax)
        for key,value in xox_spatial.named_parameters():
            if key != '1.output:gene_matrix':
                value.requires_grad = False

        Accuracies.append(res.get('best_accuracy'))
        Genes.append(g)
        NumReps.append(reps)
        ModelNames.append('MNIST')

        print(Fore.YELLOW, f"Performance on KMNIST after MNIST.", Fore.RESET)

        res = train(xox_spatial,kmnist, lr=lr_frozen, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        Genes.append(g)
        NumReps.append(reps)
        ModelNames.append('KMNIST_After_MNIST')


        Data = {'ModelName': ModelNames,'Best Accuracy': Accuracies, 'NumGenes': Genes, 'RepNum': NumReps}
        df = pd.DataFrame(data=Data)
        #df.to_csv('TransferLearningData.csv')

    