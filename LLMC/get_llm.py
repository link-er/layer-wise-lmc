import argparse
from data.utils import get_dataset
import torch
import random
import numpy as np
import config
from models.utils import get_model
from contextlib import nullcontext
import copy
from collections import OrderedDict
import pickle

def get_args():
  parser = argparse.ArgumentParser(allow_abbrev=False)
  parser.add_argument('--config_format', default='base', choices=config.registered_formats())
  args, rem_args = parser.parse_known_args()
  return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)

def get_batch(data, seq_length, batch_size, device):
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    return x, y

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

def interpolate_state_dicts(state_dict_1, state_dict_2, lambd):
    return {key: (1 - lambd) * state_dict_1[key] + lambd * state_dict_2[key]
            for key in state_dict_1.keys()}

def interpolate_layer_state_dicts(default_ind, state_dict_1, state_dict_2, layers, lambd):
    new_dict = copy.deepcopy(state_dict_1) if default_ind == 0 else copy.deepcopy(state_dict_2)
    for l in layers:
        new_dict[l] = (1 - lambd) * state_dict_1[l] + lambd * state_dict_2[l]
    return new_dict

checkpoints_path1 = "../exps/shakespeare-char/base/base_lr0.002_bs50x4_1nodes_seed=0/"
checkpoints_path2 = "../exps/shakespeare-char/base/base_lr0.002_bs50x4_1nodes_seed=42/"

if __name__ == "__main__":
    test_losses = OrderedDict()
    test_accs = OrderedDict()

    args = get_args()
    #args.dataset="shakespeare-char"
    args.n_layer=12 #2
    #args.n_head=4
    #args.n_embd=128
    args.sequence_length=256
    #args.device="cpu"
    #args.vocab_size=96

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data = get_dataset(args)

    def evaluate(model):
        device_type = 'cuda' if 'cuda' in str(args.device) else 'cpu'
        type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=torch.float16)
        data_tensor = data['val']
        sequence_length = args.sequence_length
        batch_size = args.batch_size
        device=args.device
        ctx=type_ctx
        max_num_batches=24

        loss_list_val, acc_list = [], []

        with torch.no_grad():
            for _ in range(max_num_batches):
                x, y = get_batch(data_tensor, sequence_length, batch_size, device)
                with ctx:
                    outputs = model(x, targets=y, get_logits=True)
                val_loss = outputs['loss']
                loss_list_val.append(val_loss)
                acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

            val_acc = torch.stack(acc_list).mean().item()
            val_loss = torch.stack(loss_list_val).mean().item()
            val_perplexity = 2.71828 ** val_loss

        return val_acc, val_loss, val_perplexity

    print("individual")
    for step in [0, 1500, 7500, 12000, 15000]:
        test_losses[step] = OrderedDict()
        test_accs[step] = OrderedDict()

        model1 = get_model(args).to(args.device)
        st1 = torch.load(checkpoints_path1 + "ckpt.pt" + str(step))
        for key in list(st1['model'].keys()):
          st1['model'][key.replace('_orig_mod.', '')] = st1['model'].pop(key)
        model1.load_state_dict(st1['model'])

        model2 = get_model(args).to(args.device)
        st2 = torch.load(checkpoints_path2 + "ckpt.pt" + str(step))
        for key in list(st2['model'].keys()):
            st2['model'][key.replace('_orig_mod.', '')] = st2['model'].pop(key)
        model2.load_state_dict(st1['model'])

        avg = get_model(args).to(args.device)
        avg.load_state_dict(interpolate_state_dicts(st1['model'], st2['model'], 0.5))
        avg.eval()

        # print("model1", evaluate(model1))
        # print("model2", evaluate(model2))
        # print("avg", evaluate(avg))
        test_losses[step]['full'], test_accs[step]['full'] = [], []
        val_acc, val_loss, _ = evaluate(model1)
        test_losses[step]['full'].append(val_loss)
        test_accs[step]['full'].append(val_acc)
        val_acc, val_loss, _ = evaluate(avg)
        test_losses[step]['full'].append(val_loss)
        test_accs[step]['full'].append(val_acc)
        val_acc, val_loss, _ = evaluate(model2)
        test_losses[step]['full'].append(val_loss)
        test_accs[step]['full'].append(val_acc)

        logic_layers = get_layer_blocks(st1['model'])
        for l in logic_layers:
            print(l)
            test_losses[step][l], test_accs[step][l] = [], []

            avg0 = get_model(args).to(args.device)
            avg0.load_state_dict(interpolate_layer_state_dicts(0, st1['model'], st2['model'], logic_layers[l], 0.5))
            avg0.eval()
            #print(evaluate(avg0))
            val_acc, val_loss, _ = evaluate(avg0)
            test_losses[step][l].append(val_loss)
            test_accs[step][l].append(val_acc)

            avg1 = get_model(args).to(args.device)
            avg1.load_state_dict(interpolate_layer_state_dicts(1, st1['model'], st2['model'], logic_layers[l], 0.5))
            avg1.eval()
            #print(evaluate(avg1))
            val_acc, val_loss, _ = evaluate(avg1)
            test_losses[step][l].append(val_loss)
            test_accs[step][l].append(val_acc)

    pickle.dump(test_losses, open(("test_losses.pkl"), "wb"))
    pickle.dump(test_accs, open(("test_accs.pkl"), "wb"))
