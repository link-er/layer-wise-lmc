from pathlib import Path
import numpy as np
import pickle
from models import cifar_resnet18_nonorm, cifar_vgg11
from util import get_layer_blocks, get_subset_layers
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import OrderedDict

matplotlib.rcParams.update({'font.size': 18})

networks_dir = Path("traces/cifar10_resnet18_bs256_lr0.001")
exp = 0
step = 20
net_lambda = cifar_resnet18_nonorm
loss_max = 5.0
#order = "shallow"
order = "deep"
#order = "random"
seed = 111
#order = "minimal"

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
            ax.text(j, i, round(data[i, j], 1), ha="center", va="center", color="w")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_title(title)
    ax.grid(which="minor", color='w', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    # fig.tight_layout()

def create_data_for_heatmap(hist_dict, is_loss):
    if is_loss:
        endpoints_data = (extract_column(hist_dict["full"], 0) + extract_column(hist_dict["full"], 2))/2.0
        lw_data_0, lw_data_1 = [], []
        for l in to_analyze_layers:
            lw_data_0.append(extract_column(hist_dict[l], 0) - endpoints_data)
            lw_data_1.append(extract_column(hist_dict[l], 1) - endpoints_data)
    else:
        # change accuracy to error so it is easier to understand when there is a barrier
        endpoints_data = ((100 - extract_column(hist_dict["full"], 0)) + (100 - extract_column(hist_dict["full"], 2)))/2.0
        lw_data_0, lw_data_1 = [], []
        for l in to_analyze_layers:
            lw_data_0.append((100 - extract_column(hist_dict[l], 0)) - endpoints_data)
            lw_data_1.append((100 - extract_column(hist_dict[l], 1)) - endpoints_data)
    lw_data_0 = np.array(lw_data_0)
    lw_data_1 = np.array(lw_data_1)

    return lw_data_0, lw_data_1

if __name__ == '__main__':
    train_losses = pickle.load(open(networks_dir /
                                    ("exp_" + str(exp) + str(exp+2) + "_train_losses_" + order + "_subents_step" + str(step) + ".pkl"), "rb"))
    train_accs = pickle.load(open(networks_dir /
                                  ("exp_" + str(exp) + str(exp+2) + "_train_accs_" + order + "_subents_step" + str(step) + ".pkl"), "rb"))
    test_losses = pickle.load(open(networks_dir /
                                   ("exp_" + str(exp) + str(exp+2) + "_test_losses_" + order + "_subents_step" + str(step) + ".pkl"), "rb"))
    test_accs = pickle.load(open(networks_dir /
                                 ("exp_" + str(exp) + str(exp+2) + "_test_accs_" + order + "_subents_step" + str(step) + ".pkl"), "rb"))

    logic_layers = get_layer_blocks(net_lambda().state_dict())
    to_analyze_layers = logic_layers #get_subset_layers(logic_layers)
    if order == "random":
        np.random.seed(seed)
        shuffled_keys = np.array(list(to_analyze_layers.keys()))
        np.random.shuffle(shuffled_keys)
        to_analyze_layers = OrderedDict((k, logic_layers[k]) for k in shuffled_keys)
    elif order == "minimal":
        min_ordered_keys = list(logic_layers.keys())
        min_ordered_keys = min_ordered_keys[:5] + min_ordered_keys[-5:] + min_ordered_keys[5:-5]
        to_analyze_layers = OrderedDict((k, logic_layers[k]) for k in min_ordered_keys)

    xaxis = len(train_losses["full"])
    yaxis = list(to_analyze_layers.keys())
    cbarlabel = "(relative) barrier"

    trl_data_0, trl_data_1 = create_data_for_heatmap(train_losses, is_loss=True)
    fig, axs = plt.subplots(1, 2)
    plot_heatmap(axs[0], trl_data_0, xaxis, yaxis, cbarlabel, "Interpolated into client0")
    plot_heatmap(axs[1], trl_data_1, xaxis, yaxis, cbarlabel, "Interpolated into client1")
    fig.suptitle("Train loss; cumulative averaging starting from " + order + " layers")
    plt.show()

    tra_data_0, tra_data_1 = create_data_for_heatmap(train_accs, is_loss=False)
    fig, axs = plt.subplots(1, 2)
    plot_heatmap(axs[0], tra_data_0, xaxis, yaxis, cbarlabel, "Interpolated into client0")
    plot_heatmap(axs[1], tra_data_1, xaxis, yaxis, cbarlabel, "Interpolated into client1")
    fig.suptitle("Train error; cumulative averaging starting from " + order + " layers")
    plt.show()

    tel_data_0, tel_data_1 = create_data_for_heatmap(test_losses, is_loss=True)
    fig, axs = plt.subplots(1, 2)
    plot_heatmap(axs[0], tel_data_0, xaxis, yaxis, cbarlabel, "Interpolated into client0")
    plot_heatmap(axs[1], tel_data_1, xaxis, yaxis, cbarlabel, "Interpolated into client1")
    fig.suptitle("Test loss; cumulative averaging starting from " + order + " layers")
    plt.show()

    tea_data_0, tea_data_1 = create_data_for_heatmap(test_accs, is_loss=False)
    fig, axs = plt.subplots(1, 2)
    plot_heatmap(axs[0], tea_data_0, xaxis, yaxis, cbarlabel, "Interpolated into client0")
    plot_heatmap(axs[1], tea_data_1, xaxis, yaxis, cbarlabel, "Interpolated into client1")
    fig.suptitle("Test error; cumulative averaging starting from " + order + " layers")
    plt.show()

