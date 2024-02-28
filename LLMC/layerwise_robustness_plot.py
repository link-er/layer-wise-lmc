from pathlib import Path
import numpy as np
import pickle
from models import cifar_resnet18_nonorm, cifar_vgg11, mobilenet_cifar100, mobilenet_cifar10
from util import get_layer_blocks, get_subset_layers
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams['image.cmap'] = 'Blues'

networks_dir = Path("traces/federated_noniid12_avg0_cifar10_resnet18nn_bs64_lr0.05")
exp = 0
step = 20
net_lambda = cifar_resnet18_nonorm
# on which index the data for layer replaced points
lw_mixture_index = 3 # point0 with a layer replaced to point1 layer
lw_05_mixture_index = 1 # point0 with a layer halfway to point1 layer
rbst_lw_mixture_index = 2
rbst_lw_05_mixture_index = 1
# on which index the data for the full network 0 and 1 is
full_index = 0
rbst_full_index = 0

def extract_column(dict, c_id):
    return np.array(list(dict.values()))[:, c_id]

def plot_heatmap(ax, data, xaxis, yaxis, cbarlabel, title):
    im = ax.imshow(data)
    ax.set_xticks(np.arange(xaxis), labels=np.array(list(range(xaxis))) * step)
    ax.set_xticks(np.arange(-.5, xaxis, 1), minor=True)
    ax.set_yticks(np.arange(len(yaxis)), labels=yaxis)
    ax.set_yticks(np.arange(-.5, len(yaxis), 1), minor=True)
    for i in range(len(yaxis)):
        for j in range(xaxis):
            ax.text(j, i, round(data[i, j], 1), ha="center", va="center", color="black")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_title(title)
    ax.grid(which="minor", color='w', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    #fig.tight_layout()

def create_data_for_heatmap(hist_dict, netw_ind, lw_ind, is_loss):
    if is_loss:
        lw_data = []
        if hist_dict.get("full") is None:
            l = list(to_analyze_layers.keys())[0]
            lw_data.append(extract_column(hist_dict[l], netw_ind))
        else:
            lw_data.append(extract_column(hist_dict["full"], netw_ind))
        for l in to_analyze_layers:
            lw_data.append(extract_column(hist_dict[l], lw_ind))
    else:
        lw_data = []
        if hist_dict.get("full") is None:
            l = list(to_analyze_layers.keys())[0]
            lw_data.append(100 - extract_column(hist_dict[l], netw_ind))
        else:
            lw_data.append(100 - extract_column(hist_dict["full"], netw_ind))
        for l in to_analyze_layers:
            lw_data.append(100 - extract_column(hist_dict[l], lw_ind))
    lw_data = np.array(lw_data)

    return lw_data

if __name__ == '__main__':
    train_losses = pickle.load(open(networks_dir /
                                    ("exp_" + str(exp) + "_train_losses_step" + str(step) + ".pkl"), "rb"))
    train_accs = pickle.load(open(networks_dir /
                                  ("exp_" + str(exp) + "_train_accs_step" + str(step) + ".pkl"), "rb"))
    test_losses = pickle.load(open(networks_dir /
                                   ("exp_" + str(exp) + "_test_losses_step" + str(step) + ".pkl"), "rb"))
    test_accs = pickle.load(open(networks_dir /
                                 ("exp_" + str(exp) + "_test_accs_step" + str(step) + ".pkl"), "rb"))
    rbst_train_losses = pickle.load(open(networks_dir /
                                    ("exp_" + str(exp) + "_robustness_train_losses_step" + str(step) + ".pkl"), "rb"))
    rbst_train_accs = pickle.load(open(networks_dir /
                                  ("exp_" + str(exp) + "_robustness_train_accs_step" + str(step) + ".pkl"), "rb"))
    rbst_test_losses = pickle.load(open(networks_dir /
                                   ("exp_" + str(exp) + "_robustness_test_losses_step" + str(step) + ".pkl"), "rb"))
    rbst_test_accs = pickle.load(open(networks_dir /
                                 ("exp_" + str(exp) + "_robustness_test_accs_step" + str(step) + ".pkl"), "rb"))

    logic_layers = get_layer_blocks(net_lambda().state_dict())
    to_analyze_layers = logic_layers #get_subset_layers(logic_layers)

    xaxis = len(train_losses["full"])
    yaxis = ["full"] + list(to_analyze_layers.keys())

    trl_data = create_data_for_heatmap(train_losses, full_index, lw_mixture_index, is_loss=True)
    rbst_trl_data = create_data_for_heatmap(rbst_train_losses, rbst_full_index, rbst_lw_mixture_index, is_loss=True)
    fig, axs = plt.subplots(1, 2)
    cbarlabel = "loss"
    plot_heatmap(axs[0], trl_data, xaxis, yaxis, cbarlabel, "model1 interpolation into model2")
    plot_heatmap(axs[1], rbst_trl_data, xaxis, yaxis, cbarlabel, "model1 robustness")
    fig.suptitle("Train loss")
    plt.show()

    trl_data = create_data_for_heatmap(train_losses, full_index, lw_05_mixture_index, is_loss=True)
    rbst_trl_data = create_data_for_heatmap(rbst_train_losses, rbst_full_index, rbst_lw_05_mixture_index, is_loss=True)
    fig, axs = plt.subplots(1, 2)
    cbarlabel = "loss"
    plot_heatmap(axs[0], trl_data, xaxis, yaxis, cbarlabel, "model1 alpha=0.5 interpolation into model2")
    plot_heatmap(axs[1], rbst_trl_data, xaxis, yaxis, cbarlabel, "model1 alpha=0.5 robustness")
    fig.suptitle("Train loss")
    plt.show()

    tra_data = create_data_for_heatmap(train_accs, full_index, lw_mixture_index, is_loss=False)
    rbst_tra_data = create_data_for_heatmap(rbst_train_accs, rbst_full_index, rbst_lw_mixture_index, is_loss=False)
    fig, axs = plt.subplots(1, 2)
    cbarlabel = "error"
    plot_heatmap(axs[0], tra_data, xaxis, yaxis, cbarlabel, "model1 interpolation into model2")
    plot_heatmap(axs[1], rbst_tra_data, xaxis, yaxis, cbarlabel, "model1 robustness")
    fig.suptitle("Train error")
    plt.show()

    tra_data = create_data_for_heatmap(train_accs, full_index, lw_05_mixture_index, is_loss=False)
    rbst_tra_data = create_data_for_heatmap(rbst_train_accs, rbst_full_index, rbst_lw_05_mixture_index, is_loss=False)
    fig, axs = plt.subplots(1, 2)
    cbarlabel = "error"
    plot_heatmap(axs[0], tra_data, xaxis, yaxis, cbarlabel, "model1 alpha=0.5 interpolation into model2")
    plot_heatmap(axs[1], rbst_tra_data, xaxis, yaxis, cbarlabel, "model1 alpha=0.5 robustness")
    fig.suptitle("Train error")
    plt.show()

