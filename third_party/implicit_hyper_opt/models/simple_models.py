# MIT License
#
# Copyright (c) 2018 Jonathan Lorraine, Google LLC

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .unet import UNet
from .resnet import ResNet18


class CBRStudent(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(CBRStudent, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        fcsize = 64 if num_channels == 1 else 256
        self.fc_pi = nn.Linear(fcsize, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out_pi = self.fc_pi(out)
        return out_pi


class UNetTeacher(nn.Module):
    def __init__(self, num_channels, args):
        super(UNetTeacher, self).__init__()
        self.unet = UNet(in_channels=num_channels, n_classes=1, depth=2, wf=3, padding=True, 
                         batch_norm=True, do_noise_channel=False, up_mode='upsample',use_identity_residual=False)
        self.bg_weight = args.bg
        self.min_std = args.min_std
        self.max_std = args.max_std
        self.use_exp = args.use_exp
        self.dataset = args.dataset
    def forward(self, x):
        out = self.unet(x).squeeze() # should be of shape N x H x W
#         print(out.shape)
        out = F.softmax(out.reshape(x.size(0),-1))
        out = out.reshape(x.size(0), x.size(2), x.size(3)).unsqueeze(1)
        out = out.repeat(1, 2, 1, 1) # shape N x 2 x H x W
        meshgrid_x, meshgrid_y = torch.meshgrid(torch.arange(x.size(2)),torch.arange(x.size(3)))
        mesh = torch.stack([meshgrid_x, meshgrid_y], dim=0).unsqueeze(0).cuda()
        mesh = mesh.repeat(x.size(0), 1,1,1) # shape N x 2 x H x W
        
        mean = torch.sum(out*mesh, dim=[2,3]) # shape N x 2
        
        std = self.min_std
        mask = self.bg_weight + torch.exp(torch.sum(-1*(mean.view(-1,2, 1,1) - mesh)**2 / (2*std**2), dim=1))
        return mask.unsqueeze(1)

class CBRTeacher(nn.Module):
    def __init__(self, num_channels):
        super(CBRTeacher, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        fcsize = 64 if num_channels == 1 else 256
        self.fc_cent = nn.Linear(fcsize, 2)
        self.fc_std = nn.Linear(fcsize, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        mean = x.size(2)//2 + x.size(2)//2*torch.tanh(self.fc_cent(out))
        std = 2 + 10*torch.sigmoid(self.fc_std(out))
#         print(mean.mean(dim=0), std.mean(dim=0))
        meshgrid_x, meshgrid_y = torch.meshgrid(torch.arange(x.size(2)),torch.arange(x.size(3)))
        mesh = torch.stack([meshgrid_x, meshgrid_y], dim=0).unsqueeze(0).cuda()
        mesh = mesh.repeat(x.size(0), 1,1,1)
        mask = 0.5 + torch.exp(torch.sum(-1*(mean.view(-1,2, 1,1) - mesh)**2 / (2*std**2).view(-1,2,1,1), dim=1))
        print(mean.mean(), mean.std(),std.mean(), std.std())
        return mask.unsqueeze(1).repeat(1, x.size(1), 1, 1)
       
        

class GaussianDropout(nn.Module):
    def __init__(self, dropout):
        super(GaussianDropout, self).__init__()

        self.dropout = dropout

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        # N(1, alpha)
        if self.training:
            dropout = F.sigmoid(self.dropout)
            if x.is_cuda:
                epsilon = torch.randn(x.size()).cuda() * (dropout / (1 - dropout)) + 1
            else:
                epsilon = torch.randn(x.size()) * (dropout / (1 - dropout)) + 1
            return x * epsilon
        else:
            '''
            epsilon = torch.randn(x.size()).double() * (model.dropout / (1 - model.dropout)) + 1
            if x.is_cuda:
                epsilon = epsilon.cuda()
            return x * epsilon
            '''
            return x


class BernoulliDropout(nn.Module):
    def __init__(self, dropout):
        super(BernoulliDropout, self).__init__()

        self.dropout = dropout

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        temperature = 0.5
        # N(1, alpha)
        if self.training:
            u = Variable(torch.rand(x.size()))
            if x.is_cuda:
                u = u.cuda()
            z = F.sigmoid(self.dropout) + torch.log(u / (1 - u))
            a = F.sigmoid(z / temperature)
            return x * a
        else:
            return x


class reshape(nn.Module):
    def __init__(self, size):
        super(reshape, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(-1, self.size)


class SimpleConvNet(nn.Module):
    def __init__(self, batch_norm=True, dropType='bernoulli', conv_drop1=0.0, conv_drop2=0.0, fc_drop=0.0):
        super(SimpleConvNet, self).__init__()

        self.batch_norm = batch_norm

        self.dropType = dropType
        if dropType == 'bernoulli':
            self.conv1_dropout = nn.Dropout(conv_drop1)
            self.conv2_dropout = nn.Dropout(conv_drop2)
            self.fc_dropout = nn.Dropout(fc_drop)
        elif dropType == 'gaussian':
            self.conv1_dropout = GaussianDropout(conv_drop1)
            self.conv2_dropout = GaussianDropout(conv_drop2)
            self.fc_dropout = GaussianDropout(fc_drop)

        if batch_norm:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                self.conv1_dropout,
                nn.MaxPool2d(2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                self.conv2_dropout,
                nn.MaxPool2d(2))
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                self.conv1_dropout,
                nn.MaxPool2d(2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                self.conv2_dropout,
                nn.MaxPool2d(2))

        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc_dropout(self.fc(out))
        return out


class CNN(nn.Module):
    def __init__(self, num_layers, dropout, size, weight_decay, in_channel, imsize, do_alexnet=False, num_classes=10):
        super(CNN, self).__init__()
        self.dropout = Variable(torch.FloatTensor([dropout]), requires_grad=True)
        self.weight_decay = Variable(torch.FloatTensor([weight_decay]), requires_grad=True)
        self.do_alexnet = do_alexnet
        self.num_classes = num_classes
        self.in_channel = in_channel
        self.imsize = imsize
        if self.do_alexnet:
            self.features = nn.Sequential(
                nn.Conv2d(self.in_channel, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )
            if imsize == 32:
                self.view_size = 256 * 2 * 2
            elif imsize == 28:
                self.view_size = 256
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.view_size, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, self.num_classes),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(self.in_channel, 20, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )
            if imsize == 32:
                self.view_size = 20 * 8 * 8
            elif imsize == 28:
                self.view_size = 980
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.view_size, 250),
                nn.ReLU(inplace=True),
                #nn.Dropout(),
                #nn.Linear(250, 250),
                #nn.ReLU(inplace=True),
                nn.Linear(250, self.num_classes),
            )

    def do_train(self):
        self.features.train()
        self.classifier.train()

    def do_eval(self):
        self.features.train()
        self.classifier.train()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def L2_loss(self):
        loss = 0
        for p in self.parameters():
            loss += torch.sum(torch.mul(p, p))
        return loss * (10 ** self.weight_decay)

    def all_L2_loss(self):
        loss = 0
        count = 0
        for p in self.parameters():
            #val = torch.flatten(p) - self.weight_decay[count: count + p.numel()]
            loss += torch.sum(
                torch.mul(torch.exp(self.weight_decay[count: count + p.numel()]), torch.flatten(torch.mul(p, p))))
            #loss += 1e-3 * torch.sum(torch.mul(val, val))
            count += p.numel()
        return loss


class Net(nn.Module):
    def __init__(self, num_layers, dropout, size, channel, weight_decay, num_classes=10, do_res=False,
                 do_classification=True):
        super(Net, self).__init__()
        self.dropout = Variable(torch.FloatTensor([dropout]), requires_grad=True)
        self.weight_decay = Variable(torch.FloatTensor([weight_decay]), requires_grad=True)
        self.imsize = size * size * channel
        if not do_classification: self.imsize = size * channel
        self.do_res = do_res
        l_sizes = [self.imsize, self.imsize] + [50] * 20
        network = []
        # self.Gaussian = BernoulliDropout(self.dropout)
        # network.append(nn.Dropout())
        for i in range(num_layers):
            network.append(nn.Linear(l_sizes[i], l_sizes[i + 1]))
            # network.append(self.Gaussian)
            network.append(nn.ReLU())
            #network.append(nn.Dropout())
        network.append(nn.Linear(l_sizes[num_layers], num_classes))
        self.net = nn.Sequential(*network)

    def forward(self, x):
        cur_shape = x.shape
        if not self.do_res:
            return self.net(x.view(-1, self.imsize))# .reshape(cur_shape)
        else:
            res = self.net(x.view(-1, self.imsize)).reshape(cur_shape)
            return x + res

    def do_train(self):
        self.net.train()

    def do_eval(self):
        self.net.eval()

    def L2_loss(self):
        loss = .0
        for p in self.parameters():
            loss = loss + torch.sum(torch.mul(p, p)) * torch.exp(self.weight_decay)
        return loss

    def all_L2_loss(self):
        loss = .0
        count = 0
        for p in self.parameters():
            loss = loss + torch.sum(
                torch.mul(torch.exp(self.weight_decay[count: count + p.numel()]), torch.flatten(torch.mul(p, p))))
            count += p.numel()
        return loss
