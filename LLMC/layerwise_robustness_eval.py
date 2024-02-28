# robustness comparison - take random points on the same euclidian distance as layer replacement
# and sample values of loss in them

from pathlib import Path
from torch import nn
import torch
from tqdm import *
import pickle
from collections import OrderedDict
from datasets import get_full_fed_cifar10_loaders, get_cifar10_loaders, get_cifar100_loaders
from nets_function_similarity import flatten_weights, reshape_to_state
import numpy as np

from barrier import get_layer_random_barrier
from models import cifar_resnet18_nonorm, cifar_vgg11, mobilenet_cifar100, mobilenet_cifar10
from util import get_layer_blocks, get_subset_layers

networks_dir = Path("traces/federated_avg0_cifar10_resnet18nn_bs64_lr0.05")
EPOCHS = 201
step = 20
clients_num = 2
exp = 0
random_samples = 5
criterion = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net_lambda = cifar_resnet18_nonorm
imagenet_resize = False

def get_dataloaders():
    # for single cifar
    #return get_cifar10_loaders(256, augment = False)
    #return get_cifar100_loaders(256, augment = False)
    #return get_full_omniglot_loaders(20, "data/omniglot", 256)
    # for federated cifar balanced
    return get_full_fed_cifar10_loaders(clients_num, "data/cifar10/fed_balanced", 256)
    # for federated cifar noniid
    #return get_full_fed_cifar10_loaders(clients_num, "data/cifar10/fed_noniid", 256)
    # for federated cifar noniid12
    #return get_full_fed_cifar10_loaders(clients_num, "data/cifar10/fed_noniid12", 256)

def get_states(exp, epoch):
    # for single cifar
    #s1 = torch.load(networks_dir / ("exp_" + str(exp) + "_epoch_" + str(epoch) + ".pt"))
    #s2 = torch.load(networks_dir / ("exp_" + str(7) + "_epoch_" + str(epoch) + ".pt"))
    #return s1, s2
    # for federated cifar
    s1 = torch.load(networks_dir / ("exp_" + str(exp) + "_client_0_epoch_" + str(epoch) + ".pt"))
    s2 = torch.load(networks_dir / ("exp_" + str(exp) + "_client_1_epoch_" + str(epoch) + ".pt"))
    return s1, s2

if __name__ == '__main__':
    trainloader, testloader = get_dataloaders()

    initialization1, initialization2 = get_states(exp, 0)

    logic_layers = get_layer_blocks(initialization1)
    # get only a subset of 5 layers in the network for analysis
    to_analyze_layers = logic_layers #get_subset_layers(logic_layers)

    train_losses = OrderedDict()
    train_accs = OrderedDict()
    test_losses = OrderedDict()
    test_accs = OrderedDict()

    for lname in to_analyze_layers:
        print("-----------", lname)
        ll = to_analyze_layers[lname]
        train_losses[lname] = {}
        train_accs[lname] = {}
        test_losses[lname] = {}
        test_accs[lname] = {}
        for i in tqdm(list(range(0, EPOCHS, step))):
            state1, state2 = get_states(exp, i)
            state1_ll_array = flatten_weights(OrderedDict((k, state1[k]) for k in ll))
            state2_ll_array = flatten_weights(OrderedDict((k, state2[k]) for k in ll))
            num_params = len(state1_ll_array)
            rnd_train_losses = []
            rnd_train_accs = []
            rnd_test_losses = []
            rnd_test_accs = []
            for j in range(random_samples):
                direction = np.random.normal(0, 1, num_params)
                distance = np.linalg.norm(state1_ll_array - state2_ll_array)
                center_point = state1_ll_array
                sample = center_point + distance / np.linalg.norm(direction) * direction
                rnd_layer = reshape_to_state(OrderedDict((k, state1[k]) for k in ll), sample, device)

                barriers = get_layer_random_barrier(net_lambda, ll, state1, rnd_layer, trainloader, testloader, criterion, amount=3,
                                             no_endpoints=False)

                rnd_train_losses.append([e[0] for e in barriers['train_lmc']])
                rnd_train_accs.append([e[1] for e in barriers['train_lmc']])
                rnd_test_losses.append([e[0] for e in barriers['test_lmc']])
                rnd_test_accs.append([e[1] for e in barriers['test_lmc']])

            train_losses[lname][i] = np.array(rnd_train_losses).mean(axis=0)
            train_accs[lname][i] = np.array(rnd_train_accs).mean(axis=0)
            test_losses[lname][i] = np.array(rnd_test_losses).mean(axis=0)
            test_accs[lname][i] = np.array(rnd_test_accs).mean(axis=0)

    pickle.dump(train_losses, open(networks_dir /
                                   ("exp_" + str(exp) + "_robustness_train_losses_step" + str(step) + ".pkl"), "wb"))
    pickle.dump(train_accs, open(networks_dir /
                                 ("exp_" + str(exp) + "_robustness_train_accs_step" + str(step) + ".pkl"), "wb"))
    pickle.dump(test_losses, open(networks_dir /
                                  ("exp_" + str(exp) + "_robustness_test_losses_step" + str(step) + ".pkl"), "wb"))
    pickle.dump(test_accs, open(networks_dir /
                                ("exp_" + str(exp) + "_robustness_test_accs_step" + str(step) + ".pkl"), "wb"))
