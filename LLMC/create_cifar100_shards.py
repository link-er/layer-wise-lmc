from util import noniid
from torchvision.datasets import CIFAR10, CIFAR100
import pickle

if __name__ == '__main__':
    data_path = "data/cifar100/"
    num_clients = 100
    shards_per_client = 10
    server_data = 0.0

    trainset = CIFAR100(data_path, train=True, download=True)
    testset = CIFAR100(data_path, train=False, download=True)
    dict_users_train, rand_set_all = noniid(trainset, num_clients, shards_per_client, server_data)
    dict_users_test, rand_set_all = noniid(testset, num_clients, shards_per_client, server_data, rand_set_all=rand_set_all)

    pickle.dump(dict_users_train, open(data_path + "dict_users_train.pkl", "wb"))
    pickle.dump(dict_users_test, open(data_path + "dict_users_test.pkl", "wb"))