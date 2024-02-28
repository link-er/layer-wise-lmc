from transformers import GPTNeoXForCausalLM, AutoTokenizer
from collections import OrderedDict
import copy
import os
import zipfile
import urllib
import torch
import random
import numpy as np
import pickle
from torch.nn import functional as F

WIKITEXT_DATA_PATH = "data/datasets/wikitext/"
MODEL_NAME = "410m"

def get_wikitext_data(tokenizer):
    if not os.path.exists(WIKITEXT_DATA_PATH):
        os.makedirs(WIKITEXT_DATA_PATH, exist_ok=True)
        print("downloading data and tokenizing (1-2 min)")
        raw_data_source = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip'
        urllib.request.urlretrieve(raw_data_source, os.path.join(WIKITEXT_DATA_PATH,'data.zip'))

        with zipfile.ZipFile(os.path.join(WIKITEXT_DATA_PATH, "data.zip"), 'r') as zip_ref:
            zip_ref.extractall(WIKITEXT_DATA_PATH)

    with open(os.path.join(WIKITEXT_DATA_PATH, "wikitext-103-raw/wiki.valid.raw"), 'r') as data_file:
        raw_eval_data = data_file.read()

    val_data = tokenizer(raw_eval_data, return_tensors="pt")['input_ids'][0]

    return val_data

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

def get_batch(data, seq_length, batch_size, device):
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    x = torch.stack([data[i:i + seq_length] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + seq_length] for i in ix])
    if "cuda" in torch.device(device).type:
      # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
      x = x.pin_memory().to(device, non_blocking=True)
      y = y.pin_memory().to(device, non_blocking=True)
    return x, y

def evaluate(model, data):
    sequence_length = 256
    batch_size = 50
    device = 'cuda'
    max_num_batches = 24

    loss_list_val, acc_list = [], []

    with torch.no_grad():
      for _ in range(max_num_batches):
        x, y = get_batch(data, sequence_length, batch_size, device)
        outputs = model(x)
        val_loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), y.view(-1), ignore_index=-1)
        loss_list_val.append(val_loss)
        acc_list.append((outputs.logits.argmax(-1) == y).float().mean())

      val_acc = torch.stack(acc_list).mean().item()
      val_loss = torch.stack(loss_list_val).mean().item()
      val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity


if __name__ == "__main__":
    test_losses = OrderedDict()
    test_accs = OrderedDict()

    seed = 11
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    for step in range(0, 143000, 20000):
        test_losses[step] = OrderedDict()
        test_accs[step] = OrderedDict()

        model1 = GPTNeoXForCausalLM.from_pretrained(
          "EleutherAI/pythia-"+MODEL_NAME+"-deduped",
          revision="step"+str(step),
          cache_dir="./pythia-"+MODEL_NAME+"-deduped/step"+str(step),
        )
        model1.cuda()
        model1.eval()

        tokenizer1 = AutoTokenizer.from_pretrained(
          "EleutherAI/pythia-"+MODEL_NAME+"-deduped",
          revision="step"+str(step),
          cache_dir="./pythia-"+MODEL_NAME+"-deduped/step"+str(step),
        )

        data1 = get_wikitext_data(tokenizer1)

        model2 = GPTNeoXForCausalLM.from_pretrained(
          "EleutherAI/pythia-"+MODEL_NAME,
          revision="step"+str(step),
          cache_dir="./pythia-"+MODEL_NAME+"/step"+str(step),
        )
        model2.cuda()
        model2.eval()

        tokenizer2 = AutoTokenizer.from_pretrained(
          "EleutherAI/pythia-"+MODEL_NAME,
          revision="step"+str(step),
          cache_dir="./pythia-"+MODEL_NAME+"/step"+str(step),
        )

        data2 = get_wikitext_data(tokenizer2)

        avg = copy.deepcopy(model1)
        st1 = model1.state_dict()
        st2 = model2.state_dict()
        avg.load_state_dict(interpolate_state_dicts(st1, st2, 0.5))
        avg.eval()

        # print("model1", evaluate(model1))
        # print("model2", evaluate(model2))
        # print("avg", evaluate(avg))
        test_losses[step]['full'], test_accs[step]['full'] = [], []
        val_acc, val_loss, _ = evaluate(model1, data1)
        test_losses[step]['full'].append(val_loss)
        test_accs[step]['full'].append(val_acc)
        val_acc, val_loss, _ = evaluate(avg, data1)
        test_losses[step]['full'].append(val_loss)
        test_accs[step]['full'].append(val_acc)
        val_acc, val_loss, _ = evaluate(model2, data2)
        test_losses[step]['full'].append(val_loss)
        test_accs[step]['full'].append(val_acc)

        logic_layers = get_layer_blocks(model1.state_dict())
        for l in logic_layers:
            print(l)
            test_losses[step][l], test_accs[step][l] = [], []

            avg0 = avg
            avg0.load_state_dict(interpolate_layer_state_dicts(0, st1, st2, logic_layers[l], 0.5))
            avg0.eval()
            #print(evaluate(avg0))
            val_acc, val_loss, _ = evaluate(avg0, data1)
            test_losses[step][l].append(val_loss)
            test_accs[step][l].append(val_acc)

            avg1 = avg
            avg1.load_state_dict(interpolate_layer_state_dicts(1, st1, st2, logic_layers[l], 0.5))
            avg1.eval()
            #print(evaluate(avg1))
            val_acc, val_loss, _ = evaluate(avg1, data1)
            test_losses[step][l].append(val_loss)
            test_accs[step][l].append(val_acc)

    pickle.dump(test_losses, open("test_losses.pkl", "wb"))
    pickle.dump(test_accs, open("test_accs.pkl", "wb"))
    pickle.dump(logic_layers, open("logic_layers", "wb"))