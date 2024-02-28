import copy

import numpy as np
import torch
import math
from collections import OrderedDict

def function_similarity(model, state1, state2, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net1 = model()
    net1 = net1.to(device)
    net1.load_state_dict(state1)
    net2 = model()
    net2 = net2.to(device)
    net2.load_state_dict(state2)
    softmax_predictions1, predictions1 = model_predictions(net1, dataloader)
    softmax_predictions2, predictions2 = model_predictions(net2, dataloader)

    softmax_sim = float_vectors_proximity(softmax_predictions1, softmax_predictions2)
    outputs_sim = np.count_nonzero(np.array(predictions1) == np.array(predictions2)) * 1.0 / len(predictions1)

    return softmax_sim, outputs_sim


def model_predictions(net, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net.eval()
    predictions = []
    softmax_predictions = []
    sm = torch.nn.Softmax(dim=0)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            for o in outputs:
                softmax_predictions += sm(o).detach().cpu().numpy().tolist()
            _, predicted = outputs.max(1)
            predictions += predicted.detach().cpu().numpy().tolist()

    return softmax_predictions, predictions


def flatten_weights(state_dict):
    state_array = []
    for w in list(state_dict.values()):
        state_array += w.detach().cpu().numpy().flatten().tolist()
    return np.array(state_array)

def flatten_subset_weights(state_dict, layers_to_agg):
    subset_array = []
    for l in state_dict:
        if l in set(layers_to_agg):
            subset_array += state_dict[l].detach().cpu().numpy().flatten().tolist()
        else:
            continue
    return np.array(subset_array)

def reshape_to_state(state_dict, vect, device):
    vect2dict = OrderedDict()
    i = 0
    for l in state_dict:
        param_shape = state_dict[l].size()
        param_size = math.prod(param_shape)
        vect2dict[l] = torch.tensor(np.array(vect[i:i + param_size]).reshape(param_shape), device=device)
        i += param_size
    return vect2dict

def reshape_to_state_subset(state_dict, subset_avg, device, layers_to_agg):
    vect2dict = OrderedDict()
    i = 0
    for l in state_dict:
        if l in set(layers_to_agg):
            param_shape = state_dict[l].size()
            param_size = math.prod(param_shape)
            vect2dict[l] = torch.tensor(np.array(subset_avg[i:i + param_size]).reshape(param_shape), device=device)
            i += param_size
        else:
            vect2dict[l] = copy.deepcopy(state_dict[l])
    return vect2dict

def float_vectors_proximity(vect1, vect2):
    return np.count_nonzero([abs(e1 - e2) < 1e-4 for e1, e2 in zip(vect1, vect2)]) * 1.0 / len(vect1)

