import argparse
import os
import numpy as np
import torch
import time
import data
import models
import utils
import json
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--model', default='resnet18', type=str)
    parser.add_argument('--model_path1', type=str, help='model path')
    parser.add_argument('--model_path2', type=str, help='model path')
    parser.add_argument('--n_eval', default=1024, type=int, help='#examples to evaluate on error')
    parser.add_argument('--bs', default=1024, type=int, help='batch size for error computation')
    parser.add_argument('--model_width', default=64, type=int, help='model width (# conv filters on the first layer for ResNets)')
    parser.add_argument('--alpha_step', default=0.05, type=float, help='how fine is the discretization of interpolation')
    parser.add_argument('--eval_layer_str', default='', type=str, help='which layers to evaluate on (default: take all layers; note that weights and biases will be different layers)')
    parser.add_argument('--exp_name', default='', type=str, help='experiment name (just for the sake of naming)')
    parser.add_argument('--llmc_start', default=-1, type=int, help='number of layers starting from the first one for layerwise substitution')
    parser.add_argument('--llmc_end', default=-1, type=int, help='number of layers starting from the first one for layerwise substitution')
    parser.add_argument('--log_folder', default='logs_eval', type=str)
    parser.add_argument('--interpolate_random_direction', action='store_true', help='Interpolate along a random direction')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    return parser.parse_args()


start_time = time.time()
args = get_args()

n_cls = 10 if args.dataset != 'cifar100' else 100
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
np.set_printoptions(precision=4, suppress=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

loss_f = lambda logits, y: F.cross_entropy(logits, y, reduction='mean')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model1 = models.get_model(args.model, n_cls, False, data.shapes_dict[args.dataset], args.model_width).to(device).eval()
model1.load_state_dict(torch.load(args.model_path1)['last'])
model2 = models.get_model(args.model, n_cls, False, data.shapes_dict[args.dataset], args.model_width).to(device).eval()
model2.load_state_dict(torch.load(args.model_path2)['last'])
model_int = models.get_model(args.model, n_cls, False, data.shapes_dict[args.dataset], args.model_width).to(device).eval()
model_int.load_state_dict(torch.load(args.model_path1)['last'])

eval_train_batches = data.get_loaders(args.dataset, args.n_eval, args.bs, split='train', shuffle=False,
                                      data_augm=False, drop_last=False)
eval_test_batches = data.get_loaders(args.dataset, args.n_eval, args.bs, split='test', shuffle=False,
                                     data_augm=False, drop_last=False)

### Evaluate original models (to make sure they are loaded correctly)
train_err1, train_loss1 = utils.compute_err(eval_train_batches, model1)
test_err1, test_loss1 = utils.compute_err(eval_test_batches, model1)
print('model1: train_err={:.2%} test_err={:.2%}'.format(train_err1, test_err1))
train_err2, train_loss2 = utils.compute_err(eval_train_batches, model2)
test_err2, test_loss2 = utils.compute_err(eval_test_batches, model2)
print('model2: train_err={:.2%} test_err={:.2%}'.format(train_err2, test_err2))

param_dict_int, param_dict1, param_dict2 = dict(model_int.named_parameters()), dict(model1.named_parameters()), dict(model2.named_parameters())

alpha_range = np.concatenate([np.arange(0.0, 0.0, args.alpha_step), np.arange(0.0, 1.0+args.alpha_step, args.alpha_step)])
n_layers_int = len(param_dict_int.keys()) + 1  # +1 due to full-model interpolation that goes as the first row
n_alphas = len(alpha_range)

metrics = {
    'train_err1': train_err1, 'train_loss1': train_loss1, 'test_err1': test_err1, 'test_loss1': test_loss1,
    'train_err2': train_err2, 'train_loss2': train_loss2, 'test_err2': test_err2, 'test_loss1': test_loss2,
    'param_names': [], 
    'train_err_int': np.zeros([n_layers_int, n_alphas]), 'train_loss_int': np.zeros([n_layers_int, n_alphas]), 
    'test_err_int': np.zeros([n_layers_int, n_alphas]), 'test_loss_int': np.zeros([n_layers_int, n_alphas]),
}


### Evaluate interpolations

# Full-model interpolation first 
metrics['param_names'] += ['full']
print('Interpolating full')
for i_alpha, alpha in enumerate(alpha_range):
    for i_layer, p_name in enumerate(param_dict_int.keys()):
        with torch.no_grad():
            total_perturbation_norm = torch.norm(alpha * (param_dict2[p_name].data - param_dict1[p_name].data))
            
            if args.interpolate_random_direction:
                random_within_same_distance = torch.randn_like(param_dict_int[p_name].data)
                random_within_same_distance *= torch.norm(param_dict2[p_name].data - param_dict1[p_name].data) / torch.norm(random_within_same_distance)
                param_dict_int[p_name].data = (1 - alpha) * param_dict1[p_name].data + alpha * (param_dict1[p_name].data + random_within_same_distance)
            else:  # standard interpolation
                param_dict_int[p_name].data = (1 - alpha) * param_dict1[p_name].data + alpha * param_dict2[p_name].data  # alpha=0: first model, alpha=1: second model

    if args.model == 'resnet18' and args.llmc_end == -1:
        for layer, layer1, layer2 in zip(model_int.modules(), model1.modules(), model2.modules()):
            if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
                layer.running_mean.data = (1 - alpha) * layer1.running_mean.data + alpha * layer2.running_mean.data
                layer.running_var.data = (1 - alpha) * layer1.running_var.data + alpha * layer2.running_var.data
        # utils.bn_update(eval_train_batches, model3)

    metrics['train_err_int'][0, i_alpha], metrics['train_loss_int'][0, i_alpha] = utils.compute_err(eval_train_batches, model_int)
    metrics['test_err_int'][0, i_alpha], metrics['test_loss_int'][0, i_alpha] = utils.compute_err(eval_test_batches, model_int)
    
    print('alpha={:.2f}: loss={:.3f}/{:.3f}, err={:.2%}/{:.2%} (d={:.3f})'.format(
        alpha, metrics['train_loss_int'][0, i_alpha], metrics['test_loss_int'][0, i_alpha], 
        metrics['train_err_int'][0, i_alpha], metrics['test_err_int'][0, i_alpha], total_perturbation_norm))

model_int.load_state_dict(torch.load(args.model_path1)['last'])  # revert the model


for i_layer, p_name_orig in enumerate(param_dict_int.keys()):
    if args.eval_layer_str not in p_name_orig:
        continue
    
    metrics['param_names'] += [p_name_orig]
    print(f'Interpolating {p_name_orig} ({i_layer}/{n_layers_int})')
    
    with torch.no_grad():
        for i_alpha, alpha in enumerate(alpha_range):
            # try apply interpolation also on biases if they exist in `param_dict_int`
            for p_name in [p_name_orig, p_name_orig.replace('.weight', '.bias')]:
                if p_name not in param_dict_int:
                    continue

                total_perturbation_norm = torch.norm(alpha * (param_dict2[p_name].data - param_dict1[p_name].data))
                
                if args.interpolate_random_direction:
                    random_within_same_distance = torch.randn_like(param_dict_int[p_name].data)
                    random_within_same_distance *= torch.norm(param_dict2[p_name].data - param_dict1[p_name].data) / torch.norm(random_within_same_distance)
                    param_dict_int[p_name].data = (1 - alpha) * param_dict1[p_name].data + alpha * (param_dict1[p_name].data + random_within_same_distance)
                else:  # standard interpolation
                    param_dict_int[p_name].data = (1 - alpha) * param_dict1[p_name].data + alpha * param_dict2[p_name].data  # alpha=0: first model, alpha=1: second model

            if args.model == 'resnet18' and args.llmc_end == -1:
                for layer, layer1, layer2 in zip(model_int.modules(), model1.modules(), model2.modules()):
                    if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
                        layer.running_mean.data = (1 - alpha) * layer1.running_mean.data + alpha * layer2.running_mean.data
                        layer.running_var.data = (1 - alpha) * layer1.running_var.data + alpha * layer2.running_var.data
                # utils.bn_update(eval_train_batches, model3)

            metrics['train_err_int'][i_layer+1, i_alpha], metrics['train_loss_int'][i_layer+1, i_alpha] = utils.compute_err(eval_train_batches, model_int)
            metrics['test_err_int'][i_layer+1, i_alpha], metrics['test_loss_int'][i_layer+1, i_alpha] = utils.compute_err(eval_test_batches, model_int)
            
            print('alpha={:.2f}: loss={:.3f}/{:.3f}, err={:.2%}/{:.2%} (d={:.3f})'.format(
                alpha, metrics['train_loss_int'][i_layer+1, i_alpha], metrics['test_loss_int'][i_layer+1, i_alpha], 
                metrics['train_err_int'][i_layer+1, i_alpha], metrics['test_err_int'][i_layer+1, i_alpha], total_perturbation_norm))

        for p_name in [p_name_orig, p_name_orig.replace('.weight', '.bias')]:
            if p_name not in param_dict_int:
                continue
            param_dict_int[p_name].data = param_dict1[p_name].data  # revert the change

metrics['hps'] = dict([(arg, getattr(args, arg)) for arg in vars(args)])

path = utils.get_path_interpolation(args, args.log_folder)
if not os.path.exists(args.log_folder):
    os.makedirs(args.log_folder)

np.save(path.replace('.json', '.npy'), metrics)

print('Done in {:.2f}m'.format((time.time() - start_time) / 60))


