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
import torch
import torch.nn as nn
import torch.utils.data as utils
import torch.nn.functional as F

# Uses curriculum structure
class ConvNetTeacher(nn.Module):
    def __init__(self, dataset, inner_steps):
        super(ConvNetTeacher, self).__init__()
        inpl = 2 if dataset == 'mnist' else 4
        self.layer1 = nn.Sequential(
            nn.Conv2d(inpl, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        matsize = 64 if dataset == 'mnist' else 256
        self.fc_lambda = nn.Linear(matsize,1)
        self.inner_steps = inner_steps

    def forward(self, x, itr):
        #  normalisation of itr. 
        itr = itr/self.inner_steps

        # place itr as extra channel in input image
        itrchannel = (torch.ones(x.shape[0], 1, x.shape[2],x.shape[3]).type(torch.FloatTensor)*itr).to(device)
        x = torch.cat([x, itrchannel], dim=1)

        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out_lambda = torch.sigmoid(self.fc_lambda(out))
        return out_lambda


# No curriculum structure
class ConvNetTeacher2(nn.Module):
    def __init__(self,dataset, inner_steps):
        super(ConvNetTeacher2, self).__init__()
        inpl = 1 if args.dataset == 'mnist' else 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(inpl, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        matsize = 64 if args.dataset == 'mnist' else 256
        self.fc_lambda = nn.Linear(matsize,1)

    def forward(self, x, itr):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out_lambda = torch.sigmoid(self.fc_lambda(out))
        return out_lambda

    
class ConvNetStudent(nn.Module):
    def __init__(self,dataset):
        super(ConvNetStudent, self).__init__()
        inpl = 1 if dataset == 'mnist' else 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(inpl, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        matsize = 64 if dataset == 'mnist' else 256
        self.fc_pi = nn.Linear(matsize, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out_pi = F.log_softmax(self.fc_pi(out))
        return out_pi



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(num_channels,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.weight_decay = None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out,1)
        features = out.view(out.size(0), -1)
        out = F.log_softmax(self.linear(features), dim=-1)

        if return_features:
            return out, features
        else:
            return out


def ResNet18(num_classes=10, dataset='cifar10'):
    num_channels = 3
    if dataset == 'mnist': 
        num_channels = 1
    return ResNet(BasicBlock, [2,2,2,2], num_classes, num_channels)

def ResNet34(num_classes=10, dataset='cifar10'):
    num_channels = 3
    if dataset == 'mnist': 
        num_channels = 1
    return ResNet(BasicBlock, [3,4,6,3], num_classes, num_channels=num_channels)