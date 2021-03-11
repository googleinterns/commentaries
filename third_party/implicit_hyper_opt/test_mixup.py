# MIT License
#
# Copyright (c) 2018 Jonathan Lorraine, Google LLC
#
# This script helps us train models on CIFAR with/without (already learned) blending data augmentation.

import copy
import os
import time

import pickle
import matplotlib.pyplot as plt

import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

# Local imports
import data_loaders
from models.resnet import ResNet18
from models.simple_models import CBRStudent
from aug_args_loading_utils import get_id, load_logger, save_mixup, make_parser
from utils.util import gather_flat_grad



def saver(epoch, elementary_model, elementary_optimizer, path):
    """

    :param epoch:
    :param elementary_model:
    :param elementary_optimizer:
    :param path:
    :return:
    """
    torch.save({
        'epoch': epoch,
        'elementary_model_state_dict': elementary_model.state_dict(),
        'elementary_optimizer_state_dict': elementary_optimizer.state_dict(),
    }, path + '/checkpoint.pt')


def load_mod_dl(args):
    """

    :param args:
    :return:
    """
    if args.dataset == 'cifar10':
        imsize, in_channel, num_classes = 32, 3, 10
        train_loader, val_loader, test_loader = data_loaders.load_cifar10(args.batch_size, val_split=True,
                                                                          augmentation=args.data_augmentation,
                                                                          subset=[args.train_size, args.val_size,
                                                                                  args.test_size])
    elif args.dataset == 'cifar100':
        imsize, in_channel, num_classes = 32, 3, 100
        train_loader, val_loader, test_loader = data_loaders.load_cifar100(args.batch_size, val_split=True,
                                                                           augmentation=args.data_augmentation,
                                                                           subset=[args.train_size, args.val_size,
                                                                                   args.test_size])
    elif args.dataset == 'mnist':
        imsize, in_channel, num_classes = 28, 1, 10
        num_train = 50000
        train_loader, val_loader, test_loader = data_loaders.load_mnist(args.batch_size,
                                                           subset=[args.train_size, args.val_size, args.test_size],
                                                           num_train=num_train, only_split_train=False)

    if args.model == 'resnet18':
        cnn = ResNet18(num_classes=num_classes)
    elif args.model == 'cbr':
        cnn = CBRStudent(in_channel, num_classes)
        
    # This essentially does no mixup.
    mixup_mat = -100*torch.ones([num_classes,num_classes]).cuda()

    checkpoint = None
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint)
        mixup_mat = checkpoint['mixup_grid']
        print(f"loaded mixupmat from {args.load_checkpoint}")
    
        if args.rand_mixup:
            # Randomise mixup grid
            rng = np.random.RandomState(args.seed)
            mixup_mat = rng.uniform(0.5, 1.0, (num_classes, num_classes)).astype(np.float32)
            print("Randomised the mixup mat")
        mixup_mat = torch.from_numpy(mixup_mat.reshape(num_classes,num_classes)).cuda()

    model = cnn.cuda()
    model.train()

    return model, mixup_mat, train_loader, val_loader, test_loader, checkpoint


def get_models(args):
    student, mixup_mat, train_loader, val_loader, test_loader, checkpoint = load_mod_dl(args)
    return student, mixup_mat, train_loader, val_loader, test_loader

def mixup_data(x, y, lam_rel):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    
    t_idx = torch.stack([y, y[index]], dim=1)
    lam = torch.stack([lam_rel[t_idx[i,0], t_idx[i,1]] for i in range(batch_size)])
    lam = (1  - 0.5*torch.sigmoid(lam)).view(-1,1,1,1)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam):
    return torch.mean(lam.squeeze() * F.cross_entropy(pred, y_a, reduction='none') + (1 - lam).squeeze() * F.cross_entropy(pred, y_b,reduction='none'))

def experiment(args):
    if args.do_print: 
        print(args)
    # Setup the random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model, mixup_mat, train_loader, val_loader, test_loader = get_models(args)

    # Set up the logger and the save directories

    # If we load a mixup grid, then save in a different location. Otherwise, it's a baseline run
    if args.load_checkpoint:
        args.save_dir = os.path.join(args.save_dir, 'test_aug/')
        
        # args.save_loc is for model ckpts
        args.save_loc = os.path.join(args.save_dir,get_id(args), 'test_aug_checkpoints/')
    else:
        args.save_dir = os.path.join(args.save_dir, 'test_no_aug/')
        # args.save_loc is for model ckpts
        args.save_loc = os.path.join(args.save_dir,get_id(args), 'test_no_aug_checkpoints/')

    csv_logger, _ = load_logger(args)   
    os.makedirs(args.save_loc, exist_ok=True)

    # Standard LR/schedule settings for CIFAR
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)  # [60, 120, 160]

    def train_loss_func(x, y):
        x, y = x.cuda(), y.cuda()
        if args.load_checkpoint:
            mixed_x, y_a, y_b, lam =  mixup_data(x, y, mixup_mat)
            pred = model(mixed_x)
            xentropy_loss = mixup_criterion(pred, y_a, y_b, lam)
        else:
            pred = model(x)
            xentropy_loss = F.cross_entropy(pred, y, reduction='none')

        final_loss = xentropy_loss.mean()
        return final_loss, pred

    def test(loader):
        model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        correct, total = 0., 0.
        losses = []
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                pred = model(images)
                xentropy_loss = F.cross_entropy(pred, labels)
                losses.append(xentropy_loss.item())
                xentropy_loss = F.cross_entropy(pred, labels)
                losses.append(xentropy_loss.item())

            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        avg_loss = float(np.mean(losses))
        acc = correct / total
        model.train()
        return avg_loss, acc

    init_time = time.time()
    val_loss, val_acc = test(val_loader)
    test_loss, test_acc = test(test_loader)
    if args.do_print:
        print(f"Initial Val Loss: {val_loss, val_acc}")
        print(f"Initial Test Loss: {test_loss, test_acc}")
    iteration = 0
    for epoch in range(0, args.epochs):
        reg_anneal_epoch = epoch
        xentropy_loss_avg = 0.
        total_val_loss, val_loss = 0., 0.
        correct = 0.
        total = 0.
        weight_norm, grad_norm = .0, .0

        if args.do_print:
            progress_bar = tqdm(train_loader)
        else:
            progress_bar = train_loader
        for i, (images, labels) in enumerate(progress_bar):
            if args.do_print:
                progress_bar.set_description('Epoch ' + str(epoch))

            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            xentropy_loss, pred = train_loss_func(images, labels)
            xentropy_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            xentropy_loss_avg += xentropy_loss.item()

            iteration += 1

            # Calculate running average of accuracy
            if args.do_classification:
                pred = torch.max(pred.data, 1)[1]
                total += labels.size(0)
                correct += (pred == labels.data).sum().item()
                accuracy = correct / total
            else:
                total = 1
                accuracy = 0

            if args.do_print:
                progress_bar.set_postfix(
                    train='%.4f' % (xentropy_loss_avg / (i + 1)),
                    val='%.4f' % (total_val_loss / (i + 1)),
                    acc='%.4f' % accuracy,
                    weight='%.10f' % weight_norm,
                    update='%.10f' % grad_norm
                )
            if i %  100 == 0:
                val_loss, val_acc = test(val_loader)
                test_loss, test_acc = test(test_loader)
                csv_logger.writerow({'epoch': str(epoch),
                                     'train_loss': str(xentropy_loss_avg / (i + 1)), 'train_acc': str(accuracy),
                                     'val_loss': str(val_loss), 'val_acc': str(val_acc),
                                     'test_loss': str(test_loss), 'test_acc': str(test_acc),
                                     'run_time': time.time() - init_time,
                                     'iteration': iteration})
        scheduler.step(epoch)
        train_loss = xentropy_loss_avg / (i + 1)

        only_print_final_vals = not args.do_print
        if not only_print_final_vals:
            val_loss, val_acc = test(val_loader)
            # if val_acc >= 0.99 and accuracy >= 0.99 and epoch >= 50: break
            test_loss, test_acc = test(test_loader)
            tqdm.write('val loss: {:6.4f} | val acc: {:6.4f} | test loss: {:6.4f} | test_acc: {:6.4f}'.format(
                val_loss, val_acc, test_loss, test_acc))

            csv_logger.writerow({'epoch': str(epoch),
                                 'train_loss': str(train_loss), 'train_acc': str(accuracy),
                                 'val_loss': str(val_loss), 'val_acc': str(val_acc),
                                 'test_loss': str(test_loss), 'test_acc': str(test_acc),
                                 'run_time': time.time() - init_time, 'iteration': iteration})
        else:
            if args.do_print:
                val_loss, val_acc = test(val_loader, do_test_augment=False)
                tqdm.write('val loss: {:6.4f} | val acc: {:6.4f}'.format(val_loss, val_acc))
    val_loss, val_acc = test(val_loader)
    test_loss, test_acc = test(test_loader)
    saver(args.num_finetune_epochs, model, optimizer, args.save_loc)
    return train_loss, accuracy, val_loss, val_acc, test_loss, test_acc


if __name__ == '__main__':
    args = make_parser().parse_args()
    experiment(args)