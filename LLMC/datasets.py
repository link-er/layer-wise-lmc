import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from pathlib import Path
import numpy as np
import pickle

def get_cifar10_loaders(batch_size, imagenet_resize = False, augment = True):
    if augment:
        if imagenet_resize:
            resize_transforms = [transforms.Resize(256), transforms.RandomCrop(224, padding=16)]
        else:
            resize_transforms = [transforms.RandomCrop(32, padding=4)]
        train_transform = transforms.Compose(resize_transforms + [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        if imagenet_resize:
            resize_transforms = [transforms.Resize(256), transforms.CenterCrop(224)]
        else:
            resize_transforms = []
        train_transform = transforms.Compose(resize_transforms + [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    test_transform = transforms.Compose(resize_transforms + [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = CIFAR10(root='data/cifar10', train=True, download=True, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    testset = CIFAR10(root='data/cifar10', train=False, download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, testloader

class LocalizedCifar10Dataset(Dataset):
    def __init__(self, data_path, client_id, train, transform):
        if train:
            fname = Path("cifar10train/" + str(client_id) + ".npz")
        else:
            fname = Path("cifar10test/" + str(client_id) + ".npz")
        npzfile_data = np.load(Path(data_path) / fname, allow_pickle=True)['data'].tolist()
        self.img_data = npzfile_data['x']
        self.img_labels = npzfile_data['y']
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.transform(self.img_data[idx])
        label = self.img_labels[idx]
        return image, label

def get_fed_cifar10_loaders(client_id, data_root, batch_size, imagenet_resize = False):
    if imagenet_resize:
        resize_transforms = [transforms.Resize(256), transforms.RandomCrop(224, padding=16)]
    else:
        resize_transforms = [transforms.RandomCrop(32, padding=4)]
    train_transform = transforms.Compose([transforms.ToPILImage()] + resize_transforms + [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([transforms.ToPILImage()] + resize_transforms + [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = LocalizedCifar10Dataset(data_path=data_root, client_id=client_id, train=True,
                                       transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testset = LocalizedCifar10Dataset(data_path=data_root, client_id=client_id, train=False,
                                      transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, testloader

class LocalizedCifar10DatasetFull(Dataset):
    def __init__(self, data_path, clients_num, train, transform):
        for client_id in range(clients_num):
            if train:
                fname = Path("cifar10train/" + str(client_id) + ".npz")
            else:
                fname = Path("cifar10test/" + str(client_id) + ".npz")
            npzfile_data = np.load(Path(data_path) / fname, allow_pickle=True)['data'].tolist()
            if client_id == 0:
                self.img_data = npzfile_data['x']
                self.img_labels = npzfile_data['y']
            else:
                self.img_data = np.concatenate((self.img_data, npzfile_data['x']))
                self.img_labels = np.concatenate((self.img_labels, npzfile_data['y']))
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.transform(self.img_data[idx])
        label = self.img_labels[idx]
        return image, label

def get_full_fed_cifar10_loaders(clients_num, data_root, batch_size, imagenet_resize = False):
    if imagenet_resize:
        resize_transforms = [transforms.Resize(256), transforms.CenterCrop(224)]
    else:
        resize_transforms = []
    train_transform = transforms.Compose([transforms.ToPILImage()] + resize_transforms + [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([transforms.ToPILImage()] + resize_transforms + [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = LocalizedCifar10DatasetFull(data_path=data_root, clients_num=clients_num, train=True,
                                       transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testset = LocalizedCifar10DatasetFull(data_path=data_root, clients_num=clients_num, train=False,
                                      transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, testloader

def get_cifar100_loaders(batch_size, imagenet_resize = False, augment = True):
    if augment:
        if imagenet_resize:
            resize_transforms = [transforms.Resize(256), transforms.RandomCrop(224, padding=16)]
        else:
            resize_transforms = [transforms.RandomCrop(32, padding=4)]
        train_transform = transforms.Compose(resize_transforms + [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])
    else:
        if imagenet_resize:
            resize_transforms = [transforms.Resize(256), transforms.CenterCrop(224)]
        else:
            resize_transforms = []
        train_transform = transforms.Compose(resize_transforms + [
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])
    test_transform = transforms.Compose(resize_transforms + [
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    trainset = CIFAR100(root='data/cifar100', train=True, download=True, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    testset = CIFAR100(root='data/cifar100', train=False, download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, testloader

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def get_shards_cifar100_loaders(client_id, shards_files, batch_size, imagenet_resize = False):
    if imagenet_resize:
        resize_transforms = [transforms.Resize(256), transforms.RandomCrop(224, padding=16)]
    else:
        resize_transforms = [transforms.RandomCrop(32, padding=4)]
    train_transform = transforms.Compose(resize_transforms + [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    test_transform = transforms.Compose(resize_transforms + [
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    trainset = CIFAR100('data/cifar100', train=True, download=True, transform=train_transform)
    testset = CIFAR100('data/cifar100', train=False, download=True, transform=test_transform)
    dict_users_train = pickle.load(open(shards_files + "dict_users_train.pkl", "rb"))
    dict_users_test = pickle.load(open(shards_files + "dict_users_test.pkl", "rb"))
    train_loader = DataLoader(DatasetSplit(trainset, dict_users_train[client_id]), batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    test_loader = DataLoader(DatasetSplit(testset, dict_users_test[client_id]), batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=True)

    return train_loader, test_loader
