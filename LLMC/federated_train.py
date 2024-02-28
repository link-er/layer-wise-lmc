from pathlib import Path
from torch import nn
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import *
from datasets import get_fed_cifar10_loaders, get_shards_cifar100_loaders, get_omniglot_loaders, get_fed_cifar10_loaders_platform
from models import cifar_resnet18_nonorm, cifar_lenet5, cifar_vgg11, mobilenet_cifar100, omniglot_fedavgcnn
from nets_function_similarity import flatten_weights, reshape_to_state, flatten_subset_weights, reshape_to_state_subset
from util import set_seeds
from warmup_scheduler_pytorch import WarmUpScheduler

BS = 64
LR = 0.01
EPOCHS = 100
AVG_EPOCHS = 100
local_epochs = 10
clients_num = 20
net_name = 'resnetnn'
net_lambda = cifar_resnet18_nonorm
dataset = "cifar10"
group_name = "fedavg"+str(clients_num)+"_"+dataset
layer_avg = True
# for ResNet18 without normalization layers
layers_to_agg = ["layer1.0.conv1.weight", "layer2.0.conv1.weight", "layer2.0.conv2.weight", "layer3.0.conv1.weight",
                  "layer3.0.conv2.weight", "layer4.0.conv1.weight", "layer4.0.conv2.weight", "fc.weight", "fc.bias"]
                #["layer2.0.conv1.weight", "layer2.0.conv2.weight", "layer2.0.downsample.0.weight", "layer2.1.conv1.weight", "layer2.1.conv2.weight",
                #    "layer3.0.conv1.weight", "layer3.0.conv2.weight", "layer3.0.downsample.0.weight", "layer3.1.conv1.weight", "layer3.1.conv2.weight"] 
                    #["conv1.weight", "layer1.0.conv2.weight", "layer1.1.conv1.weight", "layer1.1.conv2.weight",
                    #"layer2.0.downsample.0.weight", "layer2.1.conv1.weight", "layer2.1.conv2.weight",
                    #"layer3.0.downsample.0.weight", "layer3.1.conv1.weight", "layer3.1.conv2.weight",
                    #"layer4.0.downsample.0.weight", "layer4.1.conv1.weight", "layer4.1.conv2.weight"]
                    
                    #["conv1.weight", "layer1.0.conv1.weight", "layer1.0.conv2.weight", "layer1.1.conv1.weight", "layer1.1.conv2.weight",
                    #"layer2.0.conv1.weight", "layer2.0.conv2.weight", "layer2.0.downsample.0.weight", "layer2.1.conv1.weight", "layer2.1.conv2.weight",
                    #"layer3.0.conv1.weight", "layer3.0.conv2.weight", "layer3.0.downsample.0.weight", "layer3.1.conv1.weight", "layer3.1.conv2.weight",
                    #"layer4.0.conv1.weight", "layer4.0.conv2.weight", "layer4.0.downsample.0.weight", "layer4.1.conv1.weight", "layer4.1.conv2.weight"]
                    #,"fc.weight", "fc.bias", "layer4.1.conv2.weight", "layer4.1.conv1.weight"
                  # 
                  #"layer4.0.downsample.0.weight", "layer4.0.conv2.weight"]
#all_layers =  ["conv1.weight", "layer1.0.conv1.weight", "layer1.0.conv2.weight", "layer1.1.conv1.weight", "layer1.1.conv2.weight",
#                    "layer2.0.conv1.weight", "layer2.0.conv2.weight", "layer2.0.downsample.0.weight", "layer2.1.conv1.weight", "layer2.1.conv2.weight",
#                    "layer3.0.conv1.weight", "layer3.0.conv2.weight", "layer3.0.downsample.0.weight", "layer3.1.conv1.weight", "layer3.1.conv2.weight",
#                    "layer4.0.conv1.weight", "layer4.0.conv2.weight", "layer4.0.downsample.0.weight", "layer4.1.conv1.weight", "layer4.1.conv2.weight",
#                    "fc.weight", "fc.bias"]                
                  
# for Mobilenet
#layers_to_agg = ["layers.0.conv2.weight", "layers.1.conv1.weight", "layers.1.conv2.weight", "layers.2.conv1.weight",
#                   "layers.2.conv2.weight", "layers.3.conv1.weight", "layers.3.conv2.weight", "layers.4.conv2.weight",
#                   "layers.5.conv2.weight", "layers.6.conv2.weight", "layers.7.conv2.weight", "layers.8.conv2.weight",
#                   "layers.9.conv2.weight", "layers.10.conv2.weight", "layers.11.conv2.weight", "layers.12.conv2.weight"] #,
#                   "bn1.weight", "bn1.bias", "layers.0.bn1.weight", "layers.0.bn1.bias", "layers.0.bn2.weight", "layers.0.bn2.bias",
#                   "layers.1.bn1.weight", "layers.1.bn1.bias", "layers.1.bn2.weight", "layers.1.bn2.bias",
#                   "layers.2.bn1.weight", "layers.2.bn1.bias", "layers.2.bn2.weight", "layers.2.bn2.bias",
#                   "layers.3.bn1.weight", "layers.3.bn1.bias", "layers.3.bn2.weight", "layers.3.bn2.bias",
#                   "layers.4.bn1.weight", "layers.4.bn1.bias", "layers.4.bn2.weight", "layers.4.bn2.bias",
#                   "layers.5.bn1.weight", "layers.5.bn1.bias", "layers.5.bn2.weight", "layers.5.bn2.bias",
#                   "layers.6.bn1.weight", "layers.6.bn1.bias", "layers.6.bn2.weight", "layers.6.bn2.bias",
#                   "layers.7.bn1.weight", "layers.7.bn1.bias", "layers.7.bn2.weight", "layers.7.bn2.bias",
#                   "layers.8.bn1.weight", "layers.8.bn1.bias", "layers.8.bn2.weight", "layers.8.bn2.bias",
#                   "layers.9.bn1.weight", "layers.9.bn1.bias", "layers.9.bn2.weight", "layers.9.bn2.bias",
#                   "layers.10.bn1.weight", "layers.10.bn1.bias", "layers.10.bn2.weight", "layers.10.bn2.bias",
#                   "layers.11.bn1.weight", "layers.11.bn1.bias", "layers.11.bn2.weight", "layers.11.bn2.bias",
#                   "layers.12.bn1.weight", "layers.12.bn1.bias", "layers.12.bn2.weight", "layers.12.bn2.bias"]


def train_step(epoch, TRACE, exp, client_id, net, trainloader, testloader, optimizer, scheduler, wandb_id):
    #run = wandb.init(project="basins_fed_train", group=group_name+str(exp), reinit=True, id=wandb_id,
    #                 resume=True)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    #scheduler.step()
    #wandb.log({"train/loss_epoch": train_loss / (batch_idx + 1)})
    #wandb.log({"train/accuracy_epoch": 100. * correct / total})
    print('Train client %d; Loss: %.10f | Acc: %.3f%% (%d/%d)' % (client_id, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    #torch.save(net.state_dict(),
    #           TRACE / ("exp_" + str(exp) + "_client_" + str(client_id) + "_epoch_" + str(epoch + 1) + ".pt"))

    if (epoch + 1) % 20 == 0:
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            #wandb.log({"test/loss_epoch": test_loss / (batch_idx + 1)})
            #wandb.log({"test/accuracy_epoch": 100. * correct / total})
            print('Test client %d; Loss: %.10f | Acc: %.3f%% (%d/%d)' % (client_id, test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    #run.finish()
    return net.state_dict()

if __name__ == '__main__':
    TRACE = Path("traces")
    exp_folder = "fed"+str(clients_num)+"_pers_noniid_avg"+str(AVG_EPOCHS)+"local_epochs"+str(local_epochs)+"_"+dataset+"_" + net_name
    if not (TRACE / exp_folder).exists():
        (TRACE / exp_folder).mkdir()
    TRACE = TRACE / exp_folder

    #wandb.login(key='')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for exp in [7]: # range(0, 7, 2):
        wandb_ids, clients, optimizers, schedulers, trainloaders, testloaders, avg_weights = [], [], [], [], [], [], []
        criterion = nn.CrossEntropyLoss()
        for client_id in range(clients_num):
            print("Initialize client", client_id)
            id = wandb.util.generate_id()
            #run = wandb.init(config={'network': net_name, 'batch_size': BS, 'learning_rate': LR, 'optimizer': 'SGD',
            #                   'client_id': client_id}, project="basins_fed_train", group=group_name+str(exp),
            #           reinit=True, id=id, resume='allow')
            wandb_ids.append(id)

            set_seeds(exp)
            net = net_lambda()
            net = net.to(device)
            clients.append(net)
            torch.save(net.state_dict(), TRACE / ("exp_" + str(exp) + "_client_" + str(client_id) + "_epoch_0.pt"))

            set_seeds(exp + client_id)
            trainloader, testloader = get_fed_cifar10_loaders_platform(client_id, "data/fed20_noniid0.01/", BS)
            trainloaders.append(trainloader)
            testloaders.append(testloader)
            avg_weights.append(len(trainloader))
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
            optimizers.append(optimizer)

            #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5) 
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
            w_scheduler = WarmUpScheduler(optimizer, scheduler, len_loader=len(trainloader), warmup_steps=100,
                                      warmup_start_lr=0.0001, warmup_mode='linear')
            #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [EPOCHS * 0.5, EPOCHS * 0.75], gamma=0.1,
            #                                           last_epoch=-1)
            schedulers.append(w_scheduler)

            #run.finish()

        total_training_len = sum(avg_weights)
        avg_weights = [i/total_training_len for i in avg_weights]

        #j = 0
        for i in tqdm(list(range(EPOCHS))):
            for client_id in range(clients_num):
                train_step(i, TRACE, exp, client_id, clients[client_id], trainloaders[client_id],
                           testloaders[client_id], optimizers[client_id], schedulers[client_id], wandb_ids[client_id])
            if i < AVG_EPOCHS and (i+1) % local_epochs == 0:
                if layer_avg:
                    #layers_to_agg = [all_layers[j]]
                    #print("avg now", layers_to_agg)
                    avg = 0.0
                    for client_id in range(clients_num):
                        avg += avg_weights[client_id] * flatten_subset_weights(clients[client_id].state_dict(), layers_to_agg)
                    #avg /= clients_num
                    for client_id in range(clients_num):
                        agg_state_dict = reshape_to_state_subset(clients[client_id].state_dict(), avg,
                                                                      device, layers_to_agg)
                        clients[client_id].load_state_dict(agg_state_dict)
                    #j += 1
                    #if j>=len(all_layers):
                    #    j = 0
                else:
                    avg = 0.0
                    for client_id in range(clients_num):
                        avg += avg_weights[client_id] * flatten_weights(clients[client_id].state_dict())
                    #avg /= clients_num
                    avg_state_dict = reshape_to_state(clients[0].state_dict(), avg, device)
                    for client_id in range(clients_num):
                        clients[client_id].load_state_dict(avg_state_dict)
                
                if (i + 1) % 10 == 0:
                    avg_losses = 0.0
                    avg_accs = 0.0        
                    for client_id in range(clients_num):
                        clients[client_id].eval()
                        test_loss = 0
                        correct = 0
                        total = 0
                        with torch.no_grad():
                            for batch_idx, (inputs, targets) in enumerate(testloaders[client_id]):
                                inputs, targets = inputs.to(device), targets.to(device)
                                outputs = clients[client_id](inputs)
                                loss = criterion(outputs, targets)

                                test_loss += loss.item()
                                _, predicted = outputs.max(1)
                                total += targets.size(0)
                                correct += predicted.eq(targets).sum().item()
                            avg_losses += test_loss/(batch_idx+1)
                            avg_accs += 100.*correct/total

                    print('Average performance after averaging; Loss: %.10f | Acc: %.3f%%\n\n' % (avg_losses/clients_num, avg_accs/clients_num))

        #for client_id in range(clients_num):
        #    torch.save(clients[client_id].state_dict(),
        #               TRACE / ("exp_" + str(exp) + "_client_" + str(client_id) + ".pt"))

