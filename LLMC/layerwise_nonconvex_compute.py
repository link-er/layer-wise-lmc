# layerwise averaging computation

from pathlib import Path
from torch import nn
import torch
from tqdm import *
import pickle
from collections import OrderedDict
from datasets import get_full_fed_cifar10_loaders, get_cifar10_loaders, get_cifar100_loaders
from nets_function_similarity import flatten_weights
import numpy as np

from barrier import get_layer_barrier, get_barrier
from models import cifar_resnet18_nonorm, cifar_vgg11, mobilenet_cifar100, mobilenet_cifar10
from util import get_layer_blocks, get_subset_layers

networks_dir = Path("traces/federated_avg0_cifar10_resnet18nn_bs64_lr0.05")
# always trained epochs + 1 so the last step is displayed
EPOCHS = 201
step = 20
epochs_display = list(range(10)) + list(range(20, EPOCHS+1, 20))
clients_num = 2
exp = 0
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
    to_analyze_layers = logic_layers # get_subset_layers(logic_layers)

    train_losses = OrderedDict()
    train_accs = OrderedDict()
    test_losses = OrderedDict()
    test_accs = OrderedDict()
    distances = OrderedDict()

    print("-----------full")
    train_losses["full"] = {}
    train_accs["full"] = {}
    test_losses["full"] = {}
    test_accs["full"] = {}
    distances["full"] = {}
    # select if we want to display just all epochs with uniform step or be more precise in the beginning and
    # sparse in the end
    for i in tqdm(list(range(0, EPOCHS, step))):
        state1, state2 = get_states(exp, i)
        barriers = get_barrier(net_lambda, state1, state2, trainloader, testloader, criterion, amount=3)

        train_losses["full"][i] = [e[0] for e in barriers['train_lmc']]
        train_accs["full"][i] = [e[1] for e in barriers['train_lmc']]
        test_losses["full"][i] = [e[0] for e in barriers['test_lmc']]
        test_accs["full"][i] = [e[1] for e in barriers['test_lmc']]

        distances["full"][i] = []
        state1_ll_array = flatten_weights(state1)
        init1_ll_array = flatten_weights(initialization1)
        state2_ll_array = flatten_weights(state2)
        init2_ll_array = flatten_weights(initialization2)
        # between the clients
        distances["full"][i].append(np.linalg.norm(state1_ll_array - state2_ll_array))
        distances["full"][i].append(np.dot(state1_ll_array, state2_ll_array) /
                                (np.linalg.norm(state1_ll_array) * np.linalg.norm(state2_ll_array)))
        # from the initialization for 1
        distances["full"][i].append(np.linalg.norm(state1_ll_array - init1_ll_array))
        distances["full"][i].append(np.dot(state1_ll_array, init1_ll_array) /
                                (np.linalg.norm(state1_ll_array) * np.linalg.norm(init1_ll_array)))
        # from the initialization for 2
        distances["full"][i].append(np.linalg.norm(init2_ll_array - state2_ll_array))
        distances["full"][i].append(np.dot(init2_ll_array, state2_ll_array) /
                                (np.linalg.norm(init2_ll_array) * np.linalg.norm(state2_ll_array)))

    # do not need to compute networks themselves for each layer selection

    for lname in to_analyze_layers:
        print("-----------", lname)
        ll = to_analyze_layers[lname]
        train_losses[lname] = {}
        train_accs[lname] = {}
        test_losses[lname] = {}
        test_accs[lname] = {}
        distances[lname] = {}
        for i in tqdm(list(range(0, EPOCHS, step))):
            state1, state2 = get_states(exp, i)
            barriers = get_layer_barrier(net_lambda, ll, state1, state2, trainloader, testloader, criterion, amount=3,
                                         no_endpoints=False)

            train_losses[lname][i] = [e[0] for e in barriers['train_lmc']]
            train_accs[lname][i] = [e[1] for e in barriers['train_lmc']]
            test_losses[lname][i] = [e[0] for e in barriers['test_lmc']]
            test_accs[lname][i] = [e[1] for e in barriers['test_lmc']]

            # distances
            distances[lname][i] = []
            state1_ll_array = flatten_weights(OrderedDict((k,state1[k]) for k in ll))
            init1_ll_array = flatten_weights(OrderedDict((k,initialization1[k]) for k in ll))
            state2_ll_array = flatten_weights(OrderedDict((k,state2[k]) for k in ll))
            init2_ll_array = flatten_weights(OrderedDict((k,initialization2[k]) for k in ll))
            # between the clients
            distances[lname][i].append(np.linalg.norm(state1_ll_array - state2_ll_array))
            distances[lname][i].append(np.dot(state1_ll_array, state2_ll_array) /
                                    (np.linalg.norm(state1_ll_array) * np.linalg.norm(state2_ll_array)))
            # from the initialization for 1
            distances[lname][i].append(np.linalg.norm(state1_ll_array - init1_ll_array))
            distances[lname][i].append(np.dot(state1_ll_array, init1_ll_array) /
                                    (np.linalg.norm(state1_ll_array) * np.linalg.norm(init1_ll_array)))
            # from the initialization for 2
            distances[lname][i].append(np.linalg.norm(init2_ll_array - state2_ll_array))
            distances[lname][i].append(np.dot(init2_ll_array, state2_ll_array) /
                                    (np.linalg.norm(init2_ll_array) * np.linalg.norm(state2_ll_array)))

    pickle.dump(train_losses, open(networks_dir /
                                   ("exp_" + str(exp) + "_train_losses_step" + str(step) + ".pkl"), "wb"))
    pickle.dump(train_accs, open(networks_dir /
                                 ("exp_" + str(exp) + "_train_accs_step" + str(step) + ".pkl"), "wb"))
    pickle.dump(test_losses, open(networks_dir /
                                  ("exp_" + str(exp) + "_test_losses_step" + str(step) + ".pkl"), "wb"))
    pickle.dump(test_accs, open(networks_dir /
                                ("exp_" + str(exp) + "_test_accs_step" + str(step) + ".pkl"), "wb"))
    pickle.dump(distances, open(networks_dir /
                                ("exp_" + str(exp) + "_distances_step" + str(step) + ".pkl"), "wb"))
