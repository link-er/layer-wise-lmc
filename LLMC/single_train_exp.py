from torchvision.models import vgg11
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from pathlib import Path
from tqdm import *
import wandb
from datasets import get_cifar10_loaders, get_cifar100_loaders
from models import cifar_resnet18_nonorm, cifar_vgg11, mobilenet_cifar100, mobilenet_cifar10
from util import set_seeds, make_deterministic
from warmup_scheduler_pytorch import WarmUpScheduler

BS = 16
LR = 0.01
EPOCHS = 300
net_name = 'resnetnn'
net_lambda = cifar_resnet18_nonorm
imagenet_resize = False
data_name = 'cifar10'

if __name__ == '__main__':
    TRACE = Path("traces")
    folder_name = ("_".join([data_name, net_name, "bs", str(BS), "lr", str(LR)]))
    if not (TRACE / folder_name).exists():
        (TRACE / folder_name).mkdir()
    TRACE = TRACE / folder_name

    wandb.login(key='')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for exp in [5,7,9]:
        wandb.init(config={'network': net_name, 'dataset': data_name,
                           'batch_size': BS, 'learning_rate': LR, 'optimizer': 'SGD'},
                   project="basins_single_train", reinit=True)

        set_seeds(1)
        #make_deterministic()
        
        net = net_lambda()
        net = net.to(device)
        
        set_seeds(exp)
        trainloader, testloader = get_cifar10_loaders(BS)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        # for VGG11
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        # for MobileNet
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [EPOCHS*0.5, EPOCHS*0.75], gamma=0.1, last_epoch=-1)
        # for ResNet18
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        # for ResNet18 without norm layers
        #w_scheduler = WarmUpScheduler(optimizer, scheduler, len_loader=len(trainloader), warmup_steps=100,
        #                              warmup_start_lr=0.0001, warmup_mode='linear')

        torch.save(net.state_dict(), TRACE / ("exp_" + str(exp) + "_epoch_0.pt"))

        for i in tqdm(list(range(EPOCHS))):
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
                #w_scheduler.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            #scheduler.step()

            wandb.log({"train/loss_epoch": train_loss/(batch_idx+1)})
            wandb.log({"train/accuracy_epoch": 100.*correct/total})
            #print('Loss: %.10f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

            torch.save(net.state_dict(), TRACE / ("exp_" + str(exp) + "_epoch_" + str(i + 1) + ".pt"))

            if (i+1) % 10 == 0:
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

                    wandb.log({"test/loss_epoch": test_loss/(batch_idx+1)})
                    wandb.log({"test/accuracy_epoch": 100.*correct/total})
                    #print('Loss: %.10f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        torch.save(net.state_dict(), TRACE / ("exp_" + str(exp) + ".pt"))
