# MIT License
#
# Copyright (c) 2018 Jonathan Lorraine, Google LLC


import os
import ipdb
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

import h5py


def getSubset(data, size):
    return Subset(data, np.random.randint(0, high=len(data) - 1, size=size))


def load_mnist(batch_size, val_split=True, subset=[-1, -1, -1], num_train=50000, only_split_train=False):
    transformations = [transforms.ToTensor()]
    transformations.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(transformations)

    if val_split:
        # num_train = 50000  # Will split training set into 50,000 training and 10,000 validation images
        # Train set
        original_trainset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
        trainset = original_trainset

        trainset, valset = torch.utils.data.random_split(trainset, [num_train, len(trainset)-num_train])

        # Test set
        testset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)

        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                      num_workers=0)  # 50,000 images
        val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                    num_workers=0)  # 10,000 images
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True, pin_memory=False,
                                     num_workers=0)  # 10,000 images

        return train_dataloader, val_dataloader, test_dataloader
    else:
        trainset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)

        if subset[0] != -1:
            trainset = getSubset(trainset, subset[0])
        if subset[2] != -1:
            testset = getSubset(testset, subset[2])

        train_dataloader = DataLoader(trainset, batch_size=batch_size, pin_memory=True, shuffle=True,
                                      num_workers=0)  # 50,000 images
        test_dataloader = DataLoader(testset, batch_size=batch_size, pin_memory=True, shuffle=False,
                                     num_workers=0)  # 10,000 images
        return train_dataloader, None, test_dataloader




def load_cifar10(batch_size, num_train=45000, val_split=True, augmentation=False, subset=[-1, -1, -1],
                 only_split_train=False):
    train_transforms = []
    test_transforms = []

    if augmentation:
        train_transforms.append(transforms.RandomCrop(32, padding=4))
        train_transforms.append(transforms.RandomHorizontalFlip())

    train_transforms.append(transforms.ToTensor())
    test_transforms.append(transforms.ToTensor())

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transforms.append(normalize)
    test_transforms.append(normalize)

    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)

    if val_split:
        # num_train = 45000  # Will split training set into 45,000 training and 5,000 validation images
        # Train set
        original_trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True,
                                             transform=train_transform)
        trainset = original_trainset
        trainset, valset = torch.utils.data.random_split(trainset, [num_train, len(trainset)-num_train])
        # Test set
        testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=test_transform)

        if only_split_train:
            rand_ind = np.random.randint(0, high=len(original_trainset) - 1, size=subset[0] + subset[1])
            if subset[0] != -1:
                trainset = Subset(original_trainset, rand_ind[:subset[0]])
            if subset[2] != -1:
                testset = getSubset(testset, subset[2])
            if subset[1] != -1:
                valset = Subset(original_trainset, rand_ind[subset[0]:subset[0] + subset[1]])
        else:
            if subset[0] != -1:
                trainset = getSubset(trainset, subset[0])
            if subset[2] != -1:
                testset = getSubset(testset, subset[2])
            if subset[1] != -1:
                valset = getSubset(valset, subset[1])

        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                      num_workers=0)  # 45,000 images
        val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                    num_workers=0)  # 5,000 images
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                     num_workers=0)  # 10,000 images

        return train_dataloader, val_dataloader, test_dataloader
    else:
        trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=test_transform)

        if subset[0] != -1:
            trainset = getSubset(trainset, subset[0])
        if subset[2] != -1:
            testset = getSubset(testset, subset[2])

        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                      num_workers=2)  # 50,000 images
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                     num_workers=2)  # 10,000 images

        return train_dataloader, None, test_dataloader

    

def load_cifar100(batch_size, num_train=45000, val_split=True, augmentation=False, subset=[-1, -1, -1]):
    train_transforms = []
    test_transforms = []

    if augmentation:
        train_transforms.append(transforms.RandomCrop(32, padding=4))
        train_transforms.append(transforms.RandomHorizontalFlip())

    train_transforms.append(transforms.ToTensor())
    test_transforms.append(transforms.ToTensor())

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transforms.append(normalize)
    test_transforms.append(normalize)

    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)

    if val_split:
        # Train set
        trainset = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=train_transform)

        trainset, valset = torch.utils.data.random_split(trainset, [num_train, len(trainset)-num_train])
        # Test set
        testset = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=test_transform)

        if subset[0] != -1:
            trainset = getSubset(trainset, subset[0])
        if subset[2] != -1:
            testset = getSubset(testset, subset[2])
        if subset[1] != -1:
            valset = getSubset(valset, subset[1])

        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)  # 45,000 images
        val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=0)  # 5,000 images
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)  # 10,000 images

        return train_dataloader, val_dataloader, test_dataloader
    else:
        trainset = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=test_transform)

        if subset[0] != -1:
            trainset = getSubset(trainset, subset[0])
        if subset[2] != -1:
            testset = getSubset(testset, subset[2])

        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)  # 50,000 images
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)  # 10,000 images

        return train_dataloader, None, test_dataloader


