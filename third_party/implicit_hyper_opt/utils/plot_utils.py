# MIT License
#
# Copyright (c) 2018 Jonathan Lorraine

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
import csv
import re
import numpy as np
from cycler import cycler


do_axis_labels = False

def init_ax(fontsize=24):  # Since halving images use 25
    # Set some parameters.
    font = {'family': 'Times New Roman'}
    mpl.rc('font', **font)
    mpl.rcParams['legend.fontsize'] = fontsize
    mpl.rcParams['axes.labelsize'] = fontsize
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['axes.grid'] = True

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    return fig, ax


def setup_ax(ax, do_legend=True, alpha=0.0):
    if do_legend:
        ax.legend(fancybox=True, borderaxespad=0.0, framealpha=alpha)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.tick_params(axis='y', which='both', left=False, right=False)
    ax.grid(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    return ax


def load_from_csv(path):
    with open(path + '/epoch_h_log.csv') as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        d = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                d[name].append(row[name])
    d['epoch_h'] = [int(i) for i in d['epoch_h']]
    d['train_acc'] = [float(i) for i in d['train_acc']]
    d['test_acc'] = [float(i) for i in d['test_acc']]
    d['val_acc'] = [float(i) for i in d['val_acc']]
    d['train_loss'] = [float(i) for i in d['train_loss']]
    d['test_loss'] = [float(i) for i in d['test_loss']]
    d['val_loss'] = [float(i) for i in d['val_loss']]
    return d


def plot_accuracy_from_csv(path):
    fig, ax = init_ax()
    d = load_from_csv(path)
    ax.plot(d['epoch_h'], d['train_acc'], c='r', label='Train Accuracy')
    ax.plot(d['epoch_h'], d['val_acc'], c='g', label='Validation Accuracy')
    ax.plot(d['epoch_h'], d['test_acc'], c='b', label='Test Accuracy')

    ax = setup_ax(ax)

    if do_axis_labels:
        plt.title("Accuracies During Optimization")
    fig.savefig(f"images/accuracy_{path}.pdf", bbox_inches='tight')
    plt.close(fig)


def plot_loss_from_csv(path):
    fig, ax = init_ax()
    d = load_from_csv(path)

    ax.plot(d['epoch_h'], d['train_loss'], c='r', label='Train Loss')
    ax.plot(d['epoch_h'], d['val_loss'], c='g', label='Validation Loss')
    ax.plot(d['epoch_h'], d['test_loss'], c='b', label='Test Loss')

    ax = setup_ax(ax)

    if do_axis_labels:
        plt.title("Losses During Optimization")
    fig.savefig(f"images/loss_{path}.pdf", bbox_inches='tight')
    plt.close(fig)

def smooth_data(data, num_smooth):
    # smoothed_data = np.zeros(data.shape)
    left_fraction = 0.0
    right_fraction = 1.0 - left_fraction
    smoothed_data = [np.mean(data[max(ind - int(num_smooth*left_fraction), 0): min(ind + int(num_smooth*right_fraction), len(data))])
                     for ind in range(len(data))]
    return np.array(smoothed_data)

def plot_inversion_comparison_from_csv(path_KFAC, path_identity, path_zero, path_direct=None, alpha=1.0,
                                       do_smoothed=False):
    fig, ax = init_ax()
    d_KFAC = load_from_csv(path_KFAC)
    d_identity = load_from_csv(path_identity)
    d_zero = load_from_csv(path_zero)

    x_arg, y_arg = 'epoch_h', 'val_loss'
    def get_plot_type(ax):
        return ax.loglog

    if do_smoothed:
        num_smooth = 55
        d_KFAC[y_arg] = smooth_data(d_KFAC[y_arg], num_smooth)
        d_identity[y_arg] = smooth_data(d_identity[y_arg], num_smooth)
        d_zero[y_arg] = smooth_data(d_zero[y_arg], num_smooth)
    if y_arg[-3:] == 'acc':  # Plot error isntead
        d_KFAC[y_arg] = 1.0 - d_KFAC[y_arg]
        d_identity[y_arg] = 1.0 - d_identity[y_arg]
        d_zero[y_arg] = 1.0 - d_zero[y_arg]
    if path_direct is not None:
        d_direct = load_from_csv(path_direct)
        print(d_direct)
        get_plot_type(ax)(d_direct[x_arg], d_direct[y_arg], c='k', label='Direct', zorder=10, alpha=alpha)
    get_plot_type(ax)(d_KFAC[x_arg], d_KFAC[y_arg], c='r', label='KFAC', zorder=5, alpha=alpha)
    get_plot_type(ax)(d_identity[x_arg], d_identity[y_arg], c='g', label='Identity', zorder=1, alpha=alpha)
    get_plot_type(ax)(d_zero[x_arg], d_zero[y_arg], c='b', label='Zero', zorder=0, alpha=alpha)

    #print(d_identity[y_arg][-10:-1])
    #ax.set_ylim([0.8, 1.0])

    ax = setup_ax(ax)

    if do_axis_labels:
        plt.title("Inversion Method Comparison")
    fig.savefig(f"images/inversion_comparison_{path_identity}.pdf", bbox_inches='tight')
    plt.close(fig)

linewidth=4
def plot_overfit_comparison_from_csv(paths, alpha, name='', num_smooth=100, do_legend=True, do_yticks=True,
                                     do_simple=True):
    fig, ax = init_ax()
    datas = []
    for path in paths:
        datas += [(load_from_csv(path), path)]

    y_arg = '_acc'
    do_smoothed = True
    if do_smoothed:
        for data, path in datas:

            for arg in ['train' + y_arg, 'val' + y_arg, 'test' + y_arg]:
                if not do_simple:
                    num_smooth = 100
                data[arg] = smooth_data(data[arg], num_smooth)
                if y_arg == '_acc':
                    data[arg] = 1.0 - data[arg]


    def get_plot_type(ax):
        if y_arg == '_loss':
            return ax.loglog
        elif y_arg == '_acc':
            return ax.semilogx

    x_arg = 'epoch_h'
    for d, path in datas:
        color = 'k'
        linestyle = '-'
        #if 'CIFAR' in path:
        #    linestyle = '--'
        #elif 'MNIST' in path:
        #    linestyle = ':'

        if 'mlp' in path and 'layers=0' in path:
            if do_simple:
                color = 'r'
            else:
                color = 'm'
        elif 'mlp' in path and 'layers=1' in path:
            color = 'r'
        elif 'cnn' in path:
            color = 'g'
        elif 'alexnet' in path:
            if do_simple:
                color = 'g'  #'y'
            else:
                color = 'y'
        elif 'resnet' in path:
            color = 'b'
        markevery = max(int(float(len(d[x_arg])) / 100), 1)
        if do_simple:
            get_plot_type(ax)(d[x_arg], d['val' + y_arg], color=color, linestyle=linestyle,
                              alpha=1.0, markevery=markevery, linewidth=linewidth)
        else:
            get_plot_type(ax)(d[x_arg], d['train' + y_arg], color=color, linestyle=linestyle,
                              alpha=alpha, markevery=markevery, linewidth=linewidth)
            get_plot_type(ax)(d[x_arg], d['val' + y_arg], color=color, linestyle=linestyle,
                              alpha=alpha, markevery=markevery, marker='x', linewidth=linewidth)
            get_plot_type(ax)(d[x_arg], d['test' + y_arg], marker='o', color=color, linestyle=linestyle,
                              alpha=alpha, markevery=markevery, linewidth=linewidth)
    if do_simple:
        ax.plot([], [], label='Linear', color='r', linestyle='-', marker=',', linewidth=linewidth)
        ax.plot([], [], label='AlexNet', color='g', linestyle='-', marker=',', linewidth=linewidth)
        ax.plot([], [], label='ResNet44', color='b', linestyle='-', marker=',', linewidth=linewidth)
    else:
        ax.plot([], [], label='Training', color='k', linestyle='-', marker=',', linewidth=linewidth)
        if do_simple:
            ax.plot([], [], label='Validation', color='k', linestyle='-', linewidth=linewidth)  #, marker='x')
        else:
            ax.plot([], [], label='Validation', color='k', linestyle='-', marker='x', linewidth=linewidth)
        ax.plot([], [], label='Test', color='k', linestyle='-', marker='o', linewidth=linewidth)
        #ax.plot([], [], label='MNIST', color='k', linestyle=':', marker=',')
        #ax.plot([], [], label='CIFAR', color='k', linestyle='--', marker=',')
        linear_color = 'r'
        if not do_simple:
            linear_color = 'm'
        ax.plot([], [], label='Linear', color=linear_color, linestyle='-', marker=',', linewidth=linewidth)
        ax.plot([], [], label='1-Layer', color='r', linestyle='-', marker=',', linewidth=linewidth)
        ax.plot([], [], label='LeNet', color='g', linestyle='-', marker=',', linewidth=linewidth)
        alexnet_color = 'g'
        if not do_simple:
            alexnet_color = 'y'
        ax.plot([], [], label='AlexNet', color=alexnet_color, linestyle='-', marker=',', linewidth=linewidth)
        ax.plot([], [], label='ResNet44', color='b', linestyle='-', marker=',', linewidth=linewidth)

    # print(d_identity[y_arg][-10:-1])
    # ax.set_ylim([0.8, 1.0])
    if y_arg == '_acc':
        ax.set_ylim([0, 1.0])
    if not do_yticks:
        ax.set_yticks([])

    ax = setup_ax(ax, do_legend, alpha=0.75)

    if do_axis_labels:
        plt.title("Overfit Comparison")
    if do_simple:
        fig.savefig(f"../images/overfit_comparison_{name}_simple.pdf", bbox_inches='tight')
    else:
        fig.savefig(f"../images/overfit_comparison_{name}.pdf", bbox_inches='tight')
    plt.close(fig)


def plot_baseline_comparison_from_csv(path_ift, baseline_path):
    fig, ax = init_ax()

    d_ift = load_from_csv(path_ift)
    # TODO: Load data from baselines
    d_baseline_1 = None # load_from_csv(baseline_path)

    ax.plot(d_ift['epoch_h'], d_ift['val_loss'], c='r', label='IFT')
    # TODO: Plot baselines
    # ax.plot(d_baseline_1['epoch_h'], d_baseline_1['val_loss'], c='g', label='Baseline 1')

    ax = setup_ax(ax)

    if do_axis_labels:
        plt.title("Baseline Comparison")
    fig.savefig("images/baseline_comparison.pdf", bbox_inches='tight')
    plt.close(fig)

'''def plot_distilled_dataset(path_ift):
    fig, ax = init_ax()

    d_ift = load_from_csv(path_ift)
    # TODO: Load data from baselines
    d_baseline_1 = {}

    ax.plot(d_ift['epoch_h'], d_ift['val_loss'], c='r', label='IFT')
    # TODO: Plot baselines
    ax.plot(d_baseline_1['epoch_h'], d_baseline_1['val_loss'], c='g', label='Baseline 1')

    ax = setup_ax(ax)

    plt.title("Baseline Comparison")
    fig.savefig("images/baseline_comparison.pdf", bbox_inches='tight')
    plt.close(fig)'''


def plotResult(path):
    iteration = []
    global_step = 0
    for file in os.listdir(path):
        if file.endswith(".pkl"):
            param = str(file).split('_')
            print(param)
            iteration.append(
                [int(param[1]), float(param[2]), float(param[3]), float(param[4]), float(param[5]), float(param[6]),
                 float(param[7])])
    iteration.sort()
    i_range = [i[0] for i in iteration]
    test_loss = [i[1] for i in iteration]
    train_loss = [i[2] for i in iteration]
    test_correct = [i[3] for i in iteration]
    train_correct = [i[4] for i in iteration]
    dropout = [i[5] for i in iteration]
    f = plt.figure()
    plt.plot(i_range, test_loss)
    plt.title("test loss vs iteration")
    f.savefig("{}/test_loss.png".format(path), dpi=800)
    plt.show()
    f = plt.figure()
    plt.plot(i_range, test_correct)
    plt.title("test correct vs iteration")
    f.savefig("{}/test_correct.png".format(path), dpi=800)
    plt.show()
    f = plt.figure()
    plt.plot(i_range, train_loss)
    plt.title("train loss vs iteration")
    f.savefig("{}/train_loss.png".format(path), dpi=800)
    plt.show()
    f = plt.figure()
    plt.plot(i_range, train_correct)
    plt.title("train correct vs iteration")
    f.savefig("{}/train_correct.png".format(path), dpi=800)
    plt.show()
    f = plt.figure()
    plt.plot(i_range, dropout)
    plt.title("weight vs iteration")
    f.savefig("{}/weight.png".format(path), dpi=800)
    plt.show()
    return i_range, test_loss, train_loss, test_correct, train_correct, dropout


def plot_all(path):
    output = []
    for file in os.listdir(path):
        out = list(plotResult("{}/{}".format(path, file)))
        method = file.split('_')[1]
        if method == "KFAC":
            method = file.split('_')[2]
        out.append([method])
        output.append(out)
    n = len(out)
    f = plt.figure(figsize=(10, 10))
    cycol = iter(cm.rainbow(np.linspace(0, 1, n)))
    for out in output:
        plt.plot(out[0], out[5], c=next(cycol), label=out[-1][0])
    plt.title("weight vs iteration for different jacobian and hessian estimations")
    plt.legend()
    f.savefig("{}/all_weight.png".format(path), dpi=800)
    plt.show()
    f = plt.figure(figsize=(10, 10))
    for out in output:
        plt.plot(out[0], out[1], c=next(cycol), label=out[-1][0])
    plt.title("test loss vs iteration for different jacobian and hessian estimations")
    plt.legend()
    f.savefig("{}/all_test_loss.png".format(path), dpi=800)
    plt.show()
    f = plt.figure(figsize=(10, 10))
    for out in output:
        plt.plot(out[0], out[2], c=next(cycol), label=out[-1][0])
    plt.title("train loss vs iteration for different jacobian and hessian estimations")
    plt.legend()
    f.savefig("{}/all_train_loss.png".format(path), dpi=800)
    plt.show()
    f = plt.figure(figsize=(10, 10))
    for out in output:
        plt.plot(out[0], out[3], c=next(cycol), label=out[-1][0])
    plt.title("test correct vs iteration for different jacobian and hessian estimations")
    plt.legend()
    f.savefig("{}/all_test_correct.png".format(path), dpi=800)
    plt.show()
    f = plt.figure(figsize=(10, 10))
    for out in output:
        plt.plot(out[0], out[4], c=next(cycol), label=out[-1][0])
    plt.title("train correct vs iteration for different jacobian and hessian estimations")
    plt.legend()
    f.savefig("{}/all_train_correct.png".format(path), dpi=800)
    plt.show()


def showAllL2(path, j):
    model = torch.load(path)
    for i in range(10):
        L2 = model.weight_decay[784 * i: 784 * (i + 1)].data.cpu().numpy().reshape(28, -1)
        plt.figure()
        plt.matshow(L2)
        plt.title("l2 weight for {}th output".format(i))
        # plt.clim(-0.3, 0.3)
        plt.colorbar()
        plt.savefig("{}_l2_weight_{}.png".format(i, j), dpi=800)
        plt.close()


def showAllL2Change(path):
    j = 0
    for file in os.listdir(path):
        if file.endswith(".pkl"):
            showAllL2("{}/{}".format(path, file), j)
            j += 1


def make_path(model='mlp', hessian='identity', layers=0, size=50, valsize=50, dataset='MNIST',
              hyper_train='all_weight', hyper_value=-4):
    assert model in ['mlp', 'cnn', 'alexnet', 'resnet'], f'Model is {model}?'
    assert hessian in ['direct', 'identity', 'KFAC', 'zero'], f'Hessian is {hessian}?'
    assert size > 0 or size == -1, f'Size is {size}?'
    assert valsize > 0 or valsize == -1, f'Valsize is {valsize}?'
    assert dataset in ['MNIST', 'CIFAR10'], f'Dataset is {dataset}?'
    assert hyper_train in ['weight', 'all_weight', 'opt_data'], f'hyper_train is {hyper_train}'
    # TODO (constraint on hyper_value?)
    return f"model={model}_lrh=0.1_jacob=direct_hessian={hessian}_size={size}_valsize={valsize}_dataset={dataset}_hyper_train={hyper_train}_layers={layers}_restart=False_hyper_value={hyper_value}_"

if __name__ == "__main__":
    model = 'mlp'
    layers = 0
    size = 50
    valsize = 50
    dataset ='MNIST'  # or MNIST
    hyper_train = 'all_weight'
    hyper_value = -4

    do_plot_inversion_direct = False
    do_plot_baseline = False
    do_plot_overfit = True
    do_plot_inversion_large = False # True
    if do_plot_inversion_direct:
        size = 3500
        valsize = 3500
        path_direct_inversion = None
        if dataset == 'MNIST' and model == 'mlp' and layers == 0:
            hessian = 'direct'
            path_direct_inversion = make_path(model, hessian, layers, size, valsize, dataset, hyper_train, hyper_value)

        hessian = 'KFAC'
        path_KFAC_inversion = make_path(model, hessian, layers, size, valsize, dataset, hyper_train, hyper_value)

        #"model=mlp_lrh=0.1_jacob=direct_hessian=KFAC_size=50_valsize=50_dataset=MNIST_hyper_train=all_weight_layers=0_restart=False_hyper_value=-4_"
        hessian= 'identity'
        path_identity_inversion = make_path(model, hessian, layers, size, valsize, dataset, hyper_train, hyper_value)

        #"model=mlp_lrh=0.1_jacob=direct_hessian=identity_size=50_valsize=50_dataset=MNIST_hyper_train=all_weight_layers=0_restart=False_hyper_value=-4_"
        hessian = 'zero'
        path_zero_inversion = make_path(model, hessian, layers, size, valsize, dataset, hyper_train, hyper_value)

        plot_inversion_comparison_from_csv(path_direct=path_direct_inversion,
                                           path_KFAC=path_KFAC_inversion,
                                           path_identity=path_identity_inversion,
                                           path_zero=path_zero_inversion, alpha=0.5,
                                           do_smoothed=True)

    if do_plot_baseline:
        # TODO: Select a model/dataset/hyperparams to compare with baselines
        path_ift = path_identity_inversion  # TODO: Train a baseline IFT
        baseline_path = ""  # TODO: Establish a way to load baseline data
        plot_baseline_comparison_from_csv(path_ift, baseline_path)

    if do_plot_overfit:
        # TODO: Perhaps here we could show that for each model/dataset can perfectly overfit validation?
        hessian = 'identity'
        paths_overfit_MNIST = []
        model_list = ['mlp', 'alexnet', 'resnet']  # 'cnn',
        for dataset in ['MNIST']:
            for model in model_list:
                layer_selection = [0]
                #if model == 'mlp':
                #    layer_selection = [0, 1]
                for num_layers in layer_selection:
                    paths_overfit_MNIST += ['../' + make_path(model, hessian, num_layers, size, valsize, dataset, hyper_train, hyper_value)]

        paths_overfit_CIFAR = []
        for dataset in ['CIFAR10']:
            for model in model_list:
                layer_selection = [0]
                #if model == 'mlp':
                #    layer_selection = [0, 1]
                for num_layers in layer_selection:
                    paths_overfit_CIFAR += ['../' + make_path(model, 'identity', num_layers, size, valsize, dataset, hyper_train, hyper_value)]
        plot_overfit_comparison_from_csv(paths_overfit_MNIST, alpha=0.5, name='MNIST', do_legend=False)
        plot_overfit_comparison_from_csv(paths_overfit_CIFAR, alpha=0.5, name='CIFAR')

        model_list = ['mlp', 'cnn', 'alexnet', 'resnet']
        for dataset in ['MNIST']:
            for model in model_list:
                layer_selection = [0]
                if model == 'mlp':
                    layer_selection = [0, 1]
                for num_layers in layer_selection:
                    paths_overfit_MNIST += [
                        '../' + make_path(model, hessian, num_layers, size, valsize, dataset, hyper_train, hyper_value)]

        paths_overfit_CIFAR = []
        for dataset in ['CIFAR10']:
            for model in model_list:
                layer_selection = [0]
                if model == 'mlp':
                    layer_selection = [0, 1]
                for num_layers in layer_selection:
                    paths_overfit_CIFAR += [
                        '../' + make_path(model, 'identity', num_layers, size, valsize, dataset, hyper_train,
                                          hyper_value)]
        plot_overfit_comparison_from_csv(paths_overfit_MNIST, alpha=0.5, name='MNIST', do_legend=False, do_simple=False)
        plot_overfit_comparison_from_csv(paths_overfit_CIFAR, alpha=0.5, name='CIFAR', do_simple=False)

    if do_plot_inversion_large:
        model = 'resnet'
        dataset = 'CIFAR10'
        size = -1
        valsize = -1

        hessian = 'KFAC'
        path_KFAC_inversion = make_path(model, hessian, layers, size, valsize, dataset, hyper_train, hyper_value)

        # "model=mlp_lrh=0.1_jacob=direct_hessian=KFAC_size=50_valsize=50_dataset=MNIST_hyper_train=all_weight_layers=0_restart=False_hyper_value=-4_"
        hessian = 'identity'
        path_identity_inversion = make_path(model, hessian, layers, size, valsize, dataset, hyper_train, hyper_value)

        # "model=mlp_lrh=0.1_jacob=direct_hessian=identity_size=50_valsize=50_dataset=MNIST_hyper_train=all_weight_layers=0_restart=False_hyper_value=-4_"
        hessian = 'zero'
        path_zero_inversion = make_path(model, hessian, layers, size, valsize, dataset, hyper_train, hyper_value)

        plot_inversion_comparison_from_csv(path_direct=None,
                                           path_KFAC=path_KFAC_inversion,
                                           path_identity=path_identity_inversion,
                                           path_zero=path_zero_inversion,
                                           alpha=0.5, do_smoothed=True)

    print("Finished plotting!")
