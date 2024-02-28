from torchvision.models import vgg11
from torch import nn

def cifar_vgg11():
    net = vgg11(num_classes=10)
    return net