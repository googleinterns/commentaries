# MIT License
#
# Copyright (c) 2018 Jonathan Lorraine, Google LLC

import argparse
import copy
import os
from utils.csv_logger import CSVLogger
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch


def make_parser():
    """

    :return:
    """
    parser = argparse.ArgumentParser(description='Learning Data Augmentation Commentary Parameters with IFT')
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10', 'cifar100',], help='Choose a dataset')
    parser.add_argument('--model', default='resnet18', choices=['cbr', 'resnet18'], help='Choose a model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=-1, help='Num warmup epochs')
    parser.add_argument('--load_checkpoint', type=str, help='Path to pre-trained checkpoint to load and finetune')
    parser.add_argument('--save_dir', type=str, default='learn_blending/',
                        help='Save directory for the fine-tuned checkpoint')

    parser.add_argument('--train_size', type=int, default=-1, help='The training size')
    parser.add_argument('--val_size', type=int, default=-1, help='The training size')
    parser.add_argument('--test_size', type=int, default=-1, help='The training size')

    parser.add_argument('--data_augmentation', action='store_true', default=False,
                        help='Whether to use standard data augmentation (flips, crops)')
    parser.add_argument('--num_neumann_terms', type=int, default=1, help='The maximum number of neumann terms to use')
    parser.add_argument('--seed', type=int, default=0, help='The random seed to use')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hyperlr', type=float, default=1e-2, help='Hyper lr')
    parser.add_argument('--do_print', action='store_true', default=True,
                        help='If we should do diagnostic functions')
    parser.add_argument('--rand_mixup', action='store_true', default=False,
                        help='Rand the mixup grid')
    return parser


def get_id(args):
    """
    :param args:
    :return:
    """
    id = ''
    id += f'data:{args.dataset}'
    id += f'_model:{args.model}'
    id += f'_presetAug:{int(args.data_augmentation)}'
    id += f'_neumann:{int(args.num_neumann_terms)}'
    id += f'elemlr:{float(args.lr)}'
    id += f'_hyperlr:{float(args.hyperlr)}'
    id += f'_seed:{int(args.seed)}'
    id += f'_randmixup:{int(args.rand_mixup)}'
    return id


def load_logger(args):
    """

    :param args:
    :return:
    """
    # Setup saving information
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    sub_dir = args.save_dir + '/' + get_id(args)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    test_id = get_id(args)
    filename = os.path.join(sub_dir, 'log.csv')
    csv_logger = CSVLogger(
        fieldnames=['epoch', 'run_time', 'iteration',
                    'train_loss', 'train_acc',
                    'val_loss', 'val_acc',
                    'test_loss', 'test_acc',
                    'hypergradient_cos_diff', 'hypergradient_l2_diff'],
        filename=filename)
    return csv_logger, test_id


def save_mixup(mixmat, epoch, iteration, args):

    mixmat = 1 - 0.5*torch.sigmoid(mixmat.detach()).cpu().numpy()
    col_size = 10
    row_size = 10
    fig = plt.figure(figsize=(col_size, row_size))
    
    plt.imshow(mixmat, cmap='Reds', vmin = 0.5, vmax=1.0)

    plt.gca().set_aspect('auto')

    plt.draw()
    plt.colorbar()
    fig.savefig(f'{args.save_loc}/mixup-ep{epoch}-itr{iteration}.pdf',bbox_inches='tight')
    plt.close(fig)