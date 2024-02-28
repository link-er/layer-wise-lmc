import torch
import numpy as np
import random
import os
from collections import OrderedDict

def make_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_layer_blocks(state_dict):
    param_layers = list(state_dict.keys())
    logic_layers = OrderedDict()
    for l in param_layers:
        order_name = '-'.join(l.split('.')[:-1])
        if logic_layers.get(order_name):
            logic_layers[order_name].append(l)
        else:
            logic_layers[order_name] = [l]
    return logic_layers

def get_subset_layers(logic_layers):
    to_analyze_layers = OrderedDict()
    ids = [0, len(logic_layers) // 4, len(logic_layers) // 2, 3 * len(logic_layers) // 4, len(logic_layers) - 1]
    for ind, k in enumerate(logic_layers):
        if ind in ids:
            to_analyze_layers[k] = logic_layers[k]
    return to_analyze_layers

# taken from https://github.com/jhoon-oh/FedBABU
def noniid(dataset, num_users, shard_per_user, server_data_ratio, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users, all_idxs = {i: np.array([], dtype='int64') for i in range(num_users)}, [i for i in range(len(dataset))]

    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        assert (len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert (len(test) == len(dataset))
    assert (len(set(list(test))) == len(dataset))

    if server_data_ratio > 0.0:
        dict_users['server'] = set(np.random.choice(all_idxs, int(len(dataset) * server_data_ratio), replace=False))

    return dict_users, rand_set_all