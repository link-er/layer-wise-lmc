from pathlib import Path
from torch import nn
import torch
from tqdm import *
import pickle
from collections import OrderedDict
from datasets import get_full_fed_cifar10_loaders, get_cifar10_loaders
from nets_function_similarity import flatten_weights
import numpy as np

from barrier import get_layer_barrier, get_barrier
from models import cifar_resnet18_nonorm, cifar_vgg11
from util import get_layer_blocks, get_subset_layers

networks_dir = Path("traces/cifar10_resnet18nn_bs_256_lr_0.001")
EPOCHS = 201
step = 20
clients_num = 2
exp = 0
criterion = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net_lambda = cifar_resnet18_nonorm
imagenet_resize = False
# make groups starting from shallow layers, or deep
#order = "shallow"
order = "deep"
#order = "random"
seed = 111
#order = "minimal"

def get_dataloaders():
    # for single cifar
    return get_cifar10_loaders(256, augment = False)
    # for federated cifar balanced
    #return get_full_fed_cifar10_loaders(clients_num, "data/cifar10/fed_balanced", 256)
    # for federated cifar noniid
    #return get_full_fed_cifar10_loaders(clients_num, "data/cifar10/fed_noniid", 256)
    # for federated cifar noniid12
    #return get_full_fed_cifar10_loaders(clients_num, "data/cifar10/fed_noniid12", 256)

def get_states(exp, epoch):
    # for single cifar
    s1 = torch.load(networks_dir / ("exp_" + str(exp) + "_epoch_" + str(epoch) + ".pt"))
    s2 = torch.load(networks_dir / ("exp_" + str(exp + 2) + "_epoch_" + str(epoch) + ".pt"))
    return s1, s2
    # for federated cifar
    #s1 = torch.load(networks_dir / ("exp_" + str(exp) + "_client_0_epoch_" + str(epoch) + ".pt"))
    #s2 = torch.load(networks_dir / ("exp_" + str(exp) + "_client_1_epoch_" + str(epoch) + ".pt"))
    #return s1, s2

if __name__ == '__main__':
    trainloader, testloader = get_dataloaders()

    initialization1, initialization2 = get_states(exp, 0)

    logic_layers = get_layer_blocks(initialization1)
    # get only a subset of 5 layers in the network for analysis
    #to_analyze_layers = get_subset_layers(logic_layers)
    if order == "deep":
        logic_layers = OrderedDict((k, logic_layers[k]) for k in reversed(logic_layers))
    elif order == "random":
        np.random.seed(seed)
        shuffled_keys = np.array(list(logic_layers.keys()))
        np.random.shuffle(shuffled_keys)
        logic_layers = OrderedDict((k, logic_layers[k]) for k in shuffled_keys)
    elif order == "minimal":
        min_ordered_keys = list(logic_layers.keys())
        min_ordered_keys = min_ordered_keys[:5] + min_ordered_keys[-5:] + min_ordered_keys[5:-5]
        logic_layers = OrderedDict((k, logic_layers[k]) for k in min_ordered_keys)

    train_losses = OrderedDict()
    train_accs = OrderedDict()
    test_losses = OrderedDict()
    test_accs = OrderedDict()

    print("-----------full")
    train_losses["full"] = {}
    train_accs["full"] = {}
    test_losses["full"] = {}
    test_accs["full"] = {}
    for i in tqdm(list(range(0, EPOCHS, step))):
        state1, state2 = get_states(exp, i)
        barriers = get_barrier(net_lambda, state1, state2, trainloader, testloader, criterion, amount=3)

        train_losses["full"][i] = [e[0] for e in barriers['train_lmc']]
        train_accs["full"][i] = [e[1] for e in barriers['train_lmc']]
        test_losses["full"][i] = [e[0] for e in barriers['test_lmc']]
        test_accs["full"][i] = [e[1] for e in barriers['test_lmc']]

    # do not need to compute networks themselves for each layer selection
    current_group = []
    for lname in logic_layers:
        print("-----------", lname)
        current_group += logic_layers[lname]
        train_losses[lname] = {}
        train_accs[lname] = {}
        test_losses[lname] = {}
        test_accs[lname] = {}
        for i in tqdm(list(range(0, EPOCHS, step))):
            state1, state2 = get_states(exp, i)
            barriers = get_layer_barrier(net_lambda, current_group, state1, state2, trainloader, testloader,
                                               criterion, amount=3, no_endpoints=True)

            train_losses[lname][i] = [e[0] for e in barriers['train_lmc']]
            train_accs[lname][i] = [e[1] for e in barriers['train_lmc']]
            test_losses[lname][i] = [e[0] for e in barriers['test_lmc']]
            test_accs[lname][i] = [e[1] for e in barriers['test_lmc']]

    pickle.dump(train_losses, open(networks_dir /
                                   ("exp_" + str(exp) + "_train_losses_" + order + "_subents_step" + str(step) + ".pkl"), "wb"))
    pickle.dump(train_accs, open(networks_dir /
                                 ("exp_" + str(exp) + "_train_accs_" + order + "_subents_step" + str(step) + ".pkl"), "wb"))
    pickle.dump(test_losses, open(networks_dir /
                                  ("exp_" + str(exp) + "_test_losses_" + order + "_subents_step" + str(step) + ".pkl"), "wb"))
    pickle.dump(test_accs, open(networks_dir /
                                ("exp_" + str(exp) + "_test_accs_" + order + "_subents_step" + str(step) + ".pkl"), "wb"))
