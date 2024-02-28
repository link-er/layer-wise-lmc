#### adapted from https://github.com/rahimentezari/PermutationInvariance

import numpy as np
import torch
import time
import copy

def get_barrier(model, sd1, sd2, trainloader, testloader, criterion, amount = 3, no_endpoints = False):
    start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dict_barrier = {}
    line_distance = np.linspace(0, 1, amount)
    result_test = []
    result_train = []
    net = model()
    net = net.to(device)
    for i in range(len(line_distance)):
        if no_endpoints and (line_distance[i] == 0 or line_distance[i] == 1):
            continue
        if line_distance[i] == 0:
            net.load_state_dict(sd1)
        elif line_distance[i] == 1:
            net.load_state_dict(sd2)
        else:
            net.load_state_dict(interpolate_state_dicts(sd1, sd2, line_distance[i]))
        result_train.append(evaluate_model(net, trainloader, criterion))
        result_test.append(evaluate_model(net, testloader, criterion))

    dict_barrier['train_lmc'] = result_train
    dict_barrier['test_lmc'] = result_test
    #dict_barrier['barrier_test'] = (result_test[0][0] + result_test[-1][0])/2 - result_test[amount//2][0]
    #dict_barrier['barrier_train'] = (result_train[0][0] + result_train[-1][0])/2 - result_train[amount//2][0]
    end = time.time()
    return dict_barrier

def get_layer_barrier(model, layers, sd1, sd2, trainloader, testloader,
                                         criterion, amount = 3, no_endpoints = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dict_barrier = {}
    line_distance = np.linspace(0, 1, amount)
    result_test = []
    result_train = []
    net = model()
    net = net.to(device)
    for i in range(len(line_distance)):
        if no_endpoints and (line_distance[i] == 0 or line_distance[i] == 1):
            continue
        if line_distance[i] == 0:
            net.load_state_dict(interpolate_layer_state_dicts(1, sd1, sd2, layers, line_distance[i]))
            result_train.append(evaluate_model(net, trainloader, criterion))
            result_test.append(evaluate_model(net, testloader, criterion))
        elif line_distance[i] == 1:
            net.load_state_dict(interpolate_layer_state_dicts(0, sd1, sd2, layers, line_distance[i]))
            result_train.append(evaluate_model(net, trainloader, criterion))
            result_test.append(evaluate_model(net, testloader, criterion))
        else:
            # do we leave all the other parameters as in network1 or network2
            for default in [0,1]:
                net.load_state_dict(interpolate_layer_state_dicts(default, sd1, sd2, layers, line_distance[i]))
                result_train.append(evaluate_model(net, trainloader, criterion))
                result_test.append(evaluate_model(net, testloader, criterion))

    # 1 point with a layer fully replaced, two middle points (one with 0 default, one with 1 default), 0 point with a layer fully replaced
    dict_barrier['train_lmc'] = result_train
    dict_barrier['test_lmc'] = result_test
    return dict_barrier

def get_layer_random_barrier(model, layers, state, rnd_point, trainloader,
                             testloader, criterion, amount=3, no_endpoints=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dict_barrier = {}
    line_distance = np.linspace(0, 1, amount)
    result_test = []
    result_train = []
    net = model()
    net = net.to(device)
    for i in range(len(line_distance)):
        if no_endpoints and (line_distance[i] == 0 or line_distance[i] == 1):
            continue
        if line_distance[i] == 0:
            net.load_state_dict(state)
            result_train.append(evaluate_model(net, trainloader, criterion))
            result_test.append(evaluate_model(net, testloader, criterion))
        elif line_distance[i] == 1:
            net.load_state_dict(interpolate_rnd_layer(state, rnd_point, layers, line_distance[i]))
            result_train.append(evaluate_model(net, trainloader, criterion))
            result_test.append(evaluate_model(net, testloader, criterion))
        else:
            net.load_state_dict(interpolate_rnd_layer(state, rnd_point, layers, line_distance[i]))
            result_train.append(evaluate_model(net, trainloader, criterion))
            result_test.append(evaluate_model(net, testloader, criterion))

    # 1 point with a layer fully replaced, two middle points (one with 0 default, one with 1 default), 0 point with a layer fully replaced
    dict_barrier['train_lmc'] = result_train
    dict_barrier['test_lmc'] = result_test
    return dict_barrier

def interpolate_rnd_layer(state_dict, rnd_layer, layers, lambd):
    new_dict = copy.deepcopy(state_dict)
    for l in layers:
        new_dict[l] = (1 - lambd) * state_dict[l] + lambd * rnd_layer[l]
    return new_dict

def interpolate_state_dicts(state_dict_1, state_dict_2, lambd):
    return {key: (1 - lambd) * state_dict_1[key] + lambd * state_dict_2[key]
            for key in state_dict_1.keys()}

def interpolate_layer_state_dicts(default_ind, state_dict_1, state_dict_2, layers, lambd):
    new_dict = copy.deepcopy(state_dict_1) if default_ind == 0 else copy.deepcopy(state_dict_2)
    for l in layers:
        new_dict[l] = (1 - lambd) * state_dict_1[l] + lambd * state_dict_2[l]
    return new_dict

def evaluate_model(net, dataloader, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss/(batch_idx+1), 100.*correct/total