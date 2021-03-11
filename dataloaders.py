#     Copyright 2020 Google LLC

#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at

#         https://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.


import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as utils
from torchvision import datasets, transforms


def get_data(dataset, augment=False, evaluate=False): 

    # Smaller BS at training of teacher network because of memory limitations. 
    # Use larger BS at testing time.
    BS = 8
    if evaluate:
        BS = 64

    if dataset == 'cifar10':
        train_transform = []
        test_transform = []
        if augment:
            train_transform.append(transforms.RandomCrop(32, padding=4))
            train_transform.append(transforms.RandomHorizontalFlip())
        train_transform.extend([transforms.ToTensor(), transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])])
        test_transform.extend([transforms.ToTensor(), transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])])                                                                            

        trainset = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transforms.Compose(train_transform))

        if not evaluate:
            trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])

        testset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transforms.Compose(test_transform))

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                            shuffle=True, num_workers=0)
        if not evaluate:
            val_loader = torch.utils.data.DataLoader(valset, batch_size=BS,
                                                shuffle=True, num_workers=0)
        else:
            val_loader = None
        test_loader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                                shuffle=False, num_workers=0)


    elif dataset == 'cifar100':
        train_transform = []
        test_transform = []
        if augment:
            train_transform.append(transforms.RandomCrop(32, padding=4))
            train_transform.append(transforms.RandomHorizontalFlip())
        train_transform.extend([transforms.ToTensor(), transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])])
        test_transform.extend([transforms.ToTensor(), transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])])                                                                            

        trainset = datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transforms.Compose(train_transform))

        if not evaluate:
            trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])

        testset = datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transforms.Compose(test_transform))

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                            shuffle=True, num_workers=0)
        if not evaluate:
            val_loader = torch.utils.data.DataLoader(valset, batch_size=BS,
                                                shuffle=True, num_workers=0)
        else:
            val_loader = None
        test_loader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                                shuffle=False, num_workers=0)


    else:
        trainset = datasets.MNIST('./data', download=True, train=True,transform=transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                            ]))
        
        trainset, valset = torch.utils.data.random_split(trainset, [50000, 10000])

        testset = datasets.MNIST('./data', download=True, train=False,transform=transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                            ]))
        train_loader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=BS, 
                                                shuffle=True)
        if not evaluate:
            val_loader = torch.utils.data.DataLoader(valset, batch_size=BS,
                                                shuffle=True, num_workers=0)
        else:
            val_loader = None

        test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=BS, 
                                               shuffle=True)
    
    return train_loader, val_loader, test_loader