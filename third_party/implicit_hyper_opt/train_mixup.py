# MIT License
#
# Copyright (c) 2018 Jonathan Lorraine, Google LLC
#
# This code is adapted from https://github.com/lorraine2/implicit-hyper-opt/ to do commentary parameter learning for the augmentation experiments.
# 


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


def saver(epoch, elementary_model, elementary_optimizer, mixup, hyper_optimizer, path):
    """

    :param epoch:
    :param elementary_model:
    :param elementary_optimizer:
    :param maskingnet:
    :param reweighting_net:
    :param hyper_optimizer:
    :param path:
    :return:
    """
    torch.save({
        'epoch': epoch,
        'elementary_model_state_dict': elementary_model.state_dict(),
        'elementary_optimizer_state_dict': elementary_optimizer.state_dict(),
        'mixup_grid': mixup.detach().cpu().numpy(),
        'hyper_optimizer_state_dict': hyper_optimizer.state_dict()
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
        cnn = ResNet18(num_classes=num_classes, num_channels=in_channel)
    elif args.model == 'cbr':
        cnn = CBRStudent(in_channel, num_classes)
        
    mixup_mat = -1*torch.ones([num_classes,num_classes]).cuda()
    mixup_mat.requires_grad = True

    checkpoint = None
    if args.load_baseline_checkpoint:
        checkpoint = torch.load(args.load_baseline_checkpoint)
        cnn.load_state_dict(checkpoint['model_state_dict'])

    model = cnn.cuda()
    model.train()
    return model, mixup_mat, train_loader, val_loader, test_loader, checkpoint
    


def zero_hypergrad(get_hyper_train):
    """

    :param get_hyper_train:
    :return:
    """
    current_index = 0
    for p in get_hyper_train():
        p_num_params = np.prod(p.shape)
        if p.grad is not None:
            p.grad = p.grad * 0
        current_index += p_num_params


def store_hypergrad(get_hyper_train, total_d_val_loss_d_lambda):
    """

    :param get_hyper_train:
    :param total_d_val_loss_d_lambda:
    :return:
    """
    current_index = 0
    for p in get_hyper_train():
        p_num_params = np.prod(p.shape)
        p.grad = total_d_val_loss_d_lambda[current_index:current_index + p_num_params].view(p.shape)
        current_index += p_num_params


def neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, num_neumann_terms, model):
    preconditioner = d_val_loss_d_theta.detach()
    counter = preconditioner

    # Do the fixed point iteration to approximate the vector-inverseHessian product
    i = 0
    while i < num_neumann_terms:  # for i in range(num_neumann_terms):
        old_counter = counter

        # This increments counter to counter * (I - hessian) = counter - counter * hessian
        hessian_term = gather_flat_grad(
            grad(d_train_loss_d_w, model.parameters(), grad_outputs=counter.view(-1), retain_graph=True))
        counter = old_counter - elementary_lr * hessian_term

        preconditioner = preconditioner + counter
        i += 1
    return elementary_lr * preconditioner


def get_models(args):
    student, mixmat, train_loader, val_loader, test_loader, checkpoint = load_mod_dl(args)
    return student, mixmat, train_loader, val_loader, test_loader


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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model, mixup_mat, train_loader, val_loader, test_loader = get_models(args)

    csv_logger, _ = load_logger(args)

    # specify location for model saving
    args.save_loc = os.path.join(args.save_dir,get_id(args), 'train_aug_checkpoints')
    os.makedirs(args.save_loc, exist_ok=True)

    # Hyperparameter access functions
    def get_hyper_train():
        return [mixup_mat]

    def get_hyper_train_flat():
        return torch.cat([p.view(-1) for p in mixup_mat])

    # Setup the optimizers
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    hyper_optimizer = optim.Adam(get_hyper_train(), lr=args.hyperlr)

    def train_loss_func(x, y):
        x, y = x.cuda(), y.cuda()
        reg = 0.
        if args.num_neumann_terms >= 0:
            mixed_x, y_a, y_b, lam =  mixup_data(x, y, mixup_mat)
            pred = model(mixed_x)
            xentropy_loss = mixup_criterion(pred, y_a, y_b, lam)
        else:
            pred = model(x)
            xentropy_loss = F.cross_entropy(pred, y, reduction='none')

        final_loss = xentropy_loss.mean()
        return final_loss, pred

    def val_loss_func(x, y):
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        xentropy_loss = F.cross_entropy(pred, y)
        return xentropy_loss

    def test(loader):
        model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        correct, total = 0., 0.
        losses = []
        true, probpreds = [], []
        for i, (images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                pred = model(images)
                xentropy_loss = F.cross_entropy(pred, labels)
                losses.append(xentropy_loss.item())

            predcls = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (predcls == labels).sum().item()

        avg_loss = float(np.mean(losses))
        acc = correct / total
        model.train()
        return avg_loss, acc

    def hyper_step(elementary_lr, do_true_inverse=False):
        """Estimate the hypergradient, and take an update with it.
        """
        zero_hypergrad(get_hyper_train)
        num_weights, num_hypers = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in get_hyper_train())
        d_train_loss_d_w = torch.zeros(num_weights).cuda()
        model.train(), model.zero_grad()

        # First compute train loss on a batch
        for batch_idx, (x, y) in enumerate(train_loader):
            train_loss, _ = train_loss_func(x, y)
            optimizer.zero_grad()
            d_train_loss_d_w += gather_flat_grad(grad(train_loss, model.parameters(), create_graph=True))
            break
        optimizer.zero_grad()

        # Compute gradients of the validation loss w.r.t. the weights
        d_val_loss_d_theta, direct_grad = torch.zeros(num_weights).cuda(), torch.zeros(num_hypers).cuda()
        model.train(), model.zero_grad()
        for batch_idx, (x, y) in enumerate(val_loader):
            val_loss = val_loss_func(x, y)
            optimizer.zero_grad()
            d_val_loss_d_theta += gather_flat_grad(grad(val_loss, model.parameters(), retain_graph=False))
            break

        # Initialize the preconditioner and counter
        preconditioner = d_val_loss_d_theta
        # Neumann series to do hessian inversion
        preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr,
                                                              args.num_neumann_terms, model)

        # compute d / d lambda (partial Lv / partial w * partial Lt / partial w)
        # = (partial Lv / partial w * partial^2 Lt / (partial w partial lambda))
        indirect_grad = gather_flat_grad(
            grad(d_train_loss_d_w, get_hyper_train(), grad_outputs=preconditioner.view(-1)))

        # Direct grad is zero here due to no data augmentation for val data.
        hypergrad = direct_grad + indirect_grad

        zero_hypergrad(get_hyper_train)
        store_hypergrad(get_hyper_train, -hypergrad)
        return val_loss, hypergrad.norm()

    init_time = time.time()
    val_loss, val_acc = test(val_loader)
    test_loss, test_acc = test(test_loader)
    if args.do_print:
        print(f"Initial Val Loss: {val_loss, val_acc}")
        print(f"Initial Test Loss: {test_loss, test_acc}")
    iteration = 0

    # Main training loop.
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
        
        # Frequency of hypersteps
        num_tune_hyper = 1
        hyper_num = 0

        for i, (images, labels) in enumerate(progress_bar):
            if args.do_print:
                progress_bar.set_description('Epoch ' + str(epoch))

            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad() 

            # standard base model steps
            xentropy_loss, pred = train_loss_func(images, labels)
            xentropy_loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
            xentropy_loss_avg += xentropy_loss.item()

            # only do hyper steps if we have finished warmup epochs and we have nonneg steps in Neumann series
            if epoch > args.warmup_epochs and args.num_neumann_terms >= 0:
                # check if we are due to tune hypers
                if i % num_tune_hyper == 0:
                    
                    # Grab LR -- need it for hyper step
                    cur_lr = 1.0
                    for param_group in optimizer.param_groups:
                        cur_lr = param_group['lr']
                        break
                    
                    # get hypergrad and store it in .grad fields
                    val_loss, grad_norm = hyper_step(cur_lr)
                    # update hypers using this. 
                    hyper_optimizer.step()

                    weight_norm = get_hyper_train_flat().norm()
                    total_val_loss += val_loss.item()
                    hyper_num += 1

            iteration += 1

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total

            if args.do_print:
                progress_bar.set_postfix(
                    train='%.4f' % (xentropy_loss_avg / (i + 1)),
                    val='%.4f' % (total_val_loss / max(hyper_num, 1)),
                    acc='%.4f' % accuracy,
                    weight='%.4f' % weight_norm,
                    update='%.4f' % grad_norm
                )
            if i % (num_tune_hyper * 100) == 0:
                save_mixup(mixup_mat, epoch, iteration, args)
                val_loss, val_acc = test(val_loader)
                csv_logger.writerow({'epoch': str(epoch),
                                        'train_loss': str(xentropy_loss_avg / (i + 1)), 'train_acc': str(accuracy),
                                        'val_loss': str(val_loss), 'val_acc': str(val_acc),
                                        'test_loss': str(test_loss), 'test_acc': str(test_acc),
                                        'run_time': time.time() - init_time,
                                        'iteration': iteration})
        train_loss = xentropy_loss_avg / (i + 1)
        saver(iteration, model, optimizer, mixup_mat, hyper_optimizer, args.save_loc)
        only_print_final_vals = not args.do_print
        if not only_print_final_vals:
            val_loss, val_acc = test(val_loader)
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
    saver(iteration, model, optimizer, mixup_mat, hyper_optimizer, args.save_loc)


if __name__ == '__main__':
    args = make_parser().parse_args()
    experiment(args)