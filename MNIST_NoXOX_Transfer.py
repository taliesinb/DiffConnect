import torch
from torch import nn
import torch.nn as nn
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
    'ishape': [10, 10],
    'oshape': 10
}



###################################################################################################
## XOX TWO LAYER MLP Transfer                                                                    ##
###################################################################################################
Accuracies = []
ModelNames = []
NumReps = []
for reps in range(10):
        print(Fore.YELLOW, f"Performance on MNIST.", Fore.RESET)
        mnist_learned = nn.Sequential(
            Linear([28,28],100),
            nn.ReLU(),
            Linear(**io_params)
        )
        res = train(mnist_learned,mnist, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        NumReps.append(reps)
        ModelNames.append('MNIST_MLP')

        print(Fore.YELLOW, f"Performance on FMNIST after MNIST.", Fore.RESET)
        fmnist_mnist_transfer = copy.deepcopy(mnist_learned)
        for value in fmnist_mnist_transfer[0].parameters():
                value.requires_grad = False
        res = train(fmnist_mnist_transfer,fmnist, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        NumReps.append(reps)
        ModelNames.append('FMNIST_After_MNIST_MLP')


        print(Fore.YELLOW, f"Performance on KMNIST after MNIST.", Fore.RESET)
        kmnist_mnist_transfer = copy.deepcopy(mnist_learned)
        for value in kmnist_mnist_transfer[0].parameters():
                value.requires_grad = False
        res = train(kmnist_mnist_transfer,kmnist, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        NumReps.append(reps)
        ModelNames.append('KMNIST_After_MNIST_MLP')





        print(Fore.YELLOW, f"Performance on KMNIST.", Fore.RESET)
        kmnist_learned = nn.Sequential(
            Linear([28,28],100),
            nn.ReLU(),
            Linear(**io_params)
        )
        res = train(kmnist_learned,kmnist, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        NumReps.append(reps)
        ModelNames.append('KMNIST_MLP')

        print(Fore.YELLOW, f"Performance on FMNIST after KMNIST.", Fore.RESET)
        fmnist_kmnist_transfer = copy.deepcopy(kmnist_learned)
        for value in fmnist_kmnist_transfer[0].parameters():
                value.requires_grad = False
        res = train(fmnist_kmnist_transfer,fmnist, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        NumReps.append(reps)
        ModelNames.append('FMNIST_After_KMNIST_MLP')


        print(Fore.YELLOW, f"Performance on MNIST after KMNIST.", Fore.RESET)
        mnist_kmnist_transfer = copy.deepcopy(kmnist_learned)
        for value in mnist_kmnist_transfer[0].parameters():
                value.requires_grad = False
        res = train(mnist_kmnist_transfer,mnist, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        NumReps.append(reps)
        ModelNames.append('MNIST_After_KMNIST_MLP')






        print(Fore.YELLOW, f"Performance on FMNIST.", Fore.RESET)
        fmnist_learned = nn.Sequential(
            Linear([28,28],100),
            nn.ReLU(),
            Linear(**io_params)
        )
        res = train(fmnist_learned,fmnist, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        NumReps.append(reps)
        ModelNames.append('FMNIST_MLP')

        print(Fore.YELLOW, f"Performance on MNIST after FMNIST.", Fore.RESET)
        mnist_fmnist_transfer = copy.deepcopy(fmnist_learned)
        for value in mnist_fmnist_transfer[0].parameters():
                value.requires_grad = False
        res = train(mnist_fmnist_transfer,mnist, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        NumReps.append(reps)
        ModelNames.append('MNIST_After_FMNIST_MLP')


        print(Fore.YELLOW, f"Performance on KMNIST after FMNIST.", Fore.RESET)
        kmnist_fmnist_transfer = copy.deepcopy(fmnist_learned)
        for value in kmnist_fmnist_transfer[0].parameters():
                value.requires_grad = False
        res = train(kmnist_fmnist_transfer,kmnist, **train_params_nomax)

        Accuracies.append(res.get('best_accuracy'))
        NumReps.append(reps)
        ModelNames.append('KMNIST_After_FMNIST_MLP')

        


        Data = {'ModelName': ModelNames,'Best Accuracy': Accuracies, 'RepNum': NumReps}
        df = pd.DataFrame(data=Data)
        df.to_csv('TransferLearningData_MLP_100.csv')

    