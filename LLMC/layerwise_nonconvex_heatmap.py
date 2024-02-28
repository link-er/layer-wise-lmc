from pathlib import Path
import numpy as np
import pickle
from models import cifar_resnet18_nonorm, cifar_vgg11, mobilenet_cifar100
from util import get_layer_blocks, get_subset_layers
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams.update({'font.size': 11})

networks_dir = Path("traces/cifar10_resnet18nn_bs_64_lr_0.05")
exp = 0
step = 20
net_lambda = cifar_resnet18_nonorm
# on which index the data for layerwise mixture into 0 point and 1 point is
lw_mixture_index_0 = 1
lw_mixture_index_1 = 2
# on which index the data for the full network 0 and 1 is
full_index_0 = 0
full_index_1 = 2
full_barrier_index = 1

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
            ax.text(j, i, int(data[i, j]), ha="center", va="center", color="w")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_title(title)
    ax.grid(which="minor", color='w', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    #fig.tight_layout()

def create_data_for_heatmap(hist_dict, is_loss):
    if is_loss:
        endpoints_data = (extract_column(hist_dict["full"], full_index_0) + extract_column(hist_dict["full"], full_index_1))/2.0
        lw_data_0, lw_data_1 = [], []
        lw_data_0.append(extract_column(hist_dict["full"], full_barrier_index) - endpoints_data)
        lw_data_1.append(extract_column(hist_dict["full"], full_barrier_index) - endpoints_data)
        for l in to_analyze_layers:
            lw_data_0.append(extract_column(hist_dict[l], lw_mixture_index_0) - endpoints_data)
            lw_data_1.append(extract_column(hist_dict[l], lw_mixture_index_1) - endpoints_data)
    else:
        # change accuracy to error so it is easier to understand when there is a barrier
        endpoints_data = ((100 - extract_column(hist_dict["full"], full_index_0)) + (100 - extract_column(hist_dict["full"], full_index_1)))/2.0
        lw_data_0, lw_data_1 = [], []
        lw_data_0.append((100 - extract_column(hist_dict["full"], full_barrier_index)) - endpoints_data)
        lw_data_1.append((100 - extract_column(hist_dict["full"], full_barrier_index)) - endpoints_data)
        for l in to_analyze_layers:
            lw_data_0.append((100 - extract_column(hist_dict[l], lw_mixture_index_0)) - endpoints_data)
            lw_data_1.append((100 - extract_column(hist_dict[l], lw_mixture_index_1)) - endpoints_data)
    lw_data_0 = np.array(lw_data_0)
    lw_data_1 = np.array(lw_data_1)

    return lw_data_0, lw_data_1

if __name__ == '__main__':
    train_losses = pickle.load(open(networks_dir /
                                    ("exp_" + str(exp) + "_train_losses_step" + str(step) + ".pkl"), "rb"))
    train_accs = pickle.load(open(networks_dir /
                                  ("exp_" + str(exp) + "_train_accs_step" + str(step) + ".pkl"), "rb"))
    test_losses = pickle.load(open(networks_dir /
                                   ("exp_" + str(exp) + "_test_losses_step" + str(step) + ".pkl"), "rb"))
    test_accs = pickle.load(open(networks_dir /
                                 ("exp_" + str(exp) + "_test_accs_step" + str(step) + ".pkl"), "rb"))

    logic_layers = get_layer_blocks(net_lambda().state_dict())
    to_analyze_layers = logic_layers #get_subset_layers(logic_layers)

    xaxis = len(train_losses["full"])
    yaxis = ["full"] + list(to_analyze_layers.keys())
    cbarlabel = "(relative) barrier"

    trl_data_0, trl_data_1 = create_data_for_heatmap(train_losses, is_loss=True)
    fig, axs = plt.subplots(1, 2)
    plot_heatmap(axs[0], trl_data_0, xaxis, yaxis, cbarlabel, "Interpolated into model1")
    plot_heatmap(axs[1], trl_data_1, xaxis, yaxis, cbarlabel, "Interpolated into model2")
    fig.suptitle("Train loss")
    #plt.savefig(networks_dir / (str(exp) + "_train_losses_layerwise.pdf"))
    plt.show()

    tra_data_0, tra_data_1 = create_data_for_heatmap(train_accs, is_loss=False)
    fig, axs = plt.subplots(1, 2)
    plot_heatmap(axs[0], tra_data_0, xaxis, yaxis, cbarlabel, "Interpolated into model1")
    plot_heatmap(axs[1], tra_data_1, xaxis, yaxis, cbarlabel, "Interpolated into model2")
    fig.suptitle("Train error")
    #plt.savefig(networks_dir / (str(exp) + "_train_errors_layerwise.pdf"))
    plt.show()

    tel_data_0, tel_data_1 = create_data_for_heatmap(test_losses, is_loss=True)
    fig, axs = plt.subplots(1, 2)
    plot_heatmap(axs[0], tel_data_0, xaxis, yaxis, cbarlabel, "Interpolated into model1")
    plot_heatmap(axs[1], tel_data_1, xaxis, yaxis, cbarlabel, "Interpolated into model2")
    fig.suptitle("Test loss")
    #plt.savefig(networks_dir / (str(exp) + "_test_losses_layerwise.pdf"))
    plt.show()

    tea_data_0, tea_data_1 = create_data_for_heatmap(test_accs, is_loss=False)
    fig, axs = plt.subplots(1, 2)
    plot_heatmap(axs[0], tea_data_0, xaxis, yaxis, cbarlabel, "Interpolated into model1")
    plot_heatmap(axs[1], tea_data_1, xaxis, yaxis, cbarlabel, "Interpolated into model2")
    fig.suptitle("Test error")
    #plt.savefig(networks_dir / (str(exp) + "_test_errors_layerwise.pdf"))
    plt.show()

