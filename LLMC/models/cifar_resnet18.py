from torchvision.models import resnet18
from torch import nn
import torch

def cifar_resnet18_nonorm():
    net = resnet18(norm_layer=nn.Identity, num_classes=10)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    net.maxpool = nn.Identity()
    return net

def cifar_resnet18():
    net = resnet18(num_classes=10)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    net.maxpool = nn.Identity()
    return net