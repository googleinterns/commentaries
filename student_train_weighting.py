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
import torch.nn.functional as F
import higher
import pickle
from torchvision import datasets, transforms
import argparse 

from dataloaders import get_data
from models import * 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset mnist or cifar cifar100', type=str, default='cifar10')
parser.add_argument('--seed', help='seed', type=int, default=0)
parser.add_argument('--inner-lr', help='inner lr', type=float, default=1e-4)
parser.add_argument('--inner-steps', help='num inner steps', type=int, default=100)
parser.add_argument('--gpu', help='gpu to use', type=int, default=0)
parser.add_argument('--baseline', help='train baseline, no weights', action='store_true', default=False)
parser.add_argument('--arch', help='teacher architecture; curr uses curriculum (itr of training), nocurr does not', type=str, default='curr')
parser.add_argument('--studentarch', help='student architecture', type=str, default='cbr')
parser.add_argument('--augment', help='use augmentation or not', action='store_false', default=True)
parser.add_argument('--teachckpt', help='ckpt of trained teacher network for weighting', type=str)
args = parser.parse_args()

train_loader, val_loader, test_loader = get_data(args.dataset, augment=args.augment, evaluate=True)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_loss(student, teacher, x, y, itr, val=False):
    comm = teacher.forward(x, itr)
    pi_stud = student.forward(x)
    
    if val:
        y_loss_stud = torch.sum(F.nll_loss(pi_stud, y, reduction='none'))
    else:
        # weight the loss if during training.
        y_loss_stud = torch.sum(comm.squeeze()*F.nll_loss(pi_stud, y, reduction='none').squeeze())

    acc_stud = (pi_stud.argmax(dim=1) == y).sum().item()/len(y)
    return y_loss_stud, acc_stud


# Utility function to update lossdict
def update_lossdict(lossdict, update, action='append'):
    for k in update.keys():
        if action == 'append':
            if k in lossdict:
                lossdict[k].append(update[k])
            else:
                lossdict[k] = [update[k]]
        elif action == 'sum':
            if k in lossdict:
                lossdict[k] += update[k]
            else:
                lossdict[k] = update[k]
        else:
            raise NotImplementedError
    return lossdict


# Simple function to train either a standard clf model (just a student) or a student with commentaries also.
# Helps assess rate of learning over several epochs.
def baseline(train_dl, test_dl, stud_state_dict=None, teach_state_dict=None):
    torch.manual_seed(args.seed)

    stud_train_ld = {}
    stud_val_ld = {}
   
    if args.arch =='curr':
        teacher = ConvNetTeacher().to(device)
    elif args.arch == 'curr2':
        teacher = ConvNetTeacher2().to(device)
    else:
        raise NotImplementedError
    
    if teach_state_dict is not None:
        teacher.load_state_dict(teach_state_dict)
 
    # we have batch stats available so use train mode
    teacher.train()
    
    inpl = 1 if args.dataset == 'mnist' else 3
    num_classes = 100 if args.dataset == 'cifar100' else 10

    if args.studentarch == 'cbr':
        student = ConvNetStudent(args.dataset).to(device)
    elif args.studentarch == 'resnet18':
        student = ResNet18(args.dataset).to(device)
    elif args.studentarch == 'resnet34':
        student = ResNet34(args.dataset).to(device)
    
    if stud_state_dict is not None:
        try:
            student.load_state_dict(stud_state_dict)
        except Exception as e:
            print('Could not load state dict. Starting from fresh model.')
    student.train()
  
    stud_optim = torch.optim.Adam(student.parameters(), lr=args.inner_lr)
    
    numep = 5

    for ep in range(numep):
        ld = {
                'stud_loss' : [],
                'stud_acc' : []
        }
        for i, (x,y) in enumerate(train_dl):
            student.train()
            x = x.to(device)
            y = y.to(device)
            comm_steps = ep*len(train_dl) + i
            
            # Calling val=True means we don't apply any weighting.
            if args.baseline:
                y_loss, acc = get_loss(student, teacher, x, y, comm_steps, val=True)
            else:
                y_loss, acc = get_loss(student, teacher, x, y, comm_steps, val=False)
            
            loss = y_loss
            stud_optim.zero_grad() 
            loss.backward()
            stud_optim.step()

            # logging
            ld['stud_loss'].append(loss.item())
            ld['stud_acc'].append(acc)

            do_eval = False
            if ep ==0 and i ==0: 
                do_eval=True
            eval_pts = [int(1/4*len(train_dl)), int(1/2*len(train_dl)), int(3/4*len(train_dl))]
            if i in eval_pts: 
                do_eval = True
            if do_eval:
                print('Train student: Epoch %d, Step %d' % (ep, i))
                tld = eval_student(student, test_dl)
                stud_val_ld = update_lossdict(stud_val_ld, tld)

                train_ep = eval_student(student, train_dl)
                stud_train_ld = update_lossdict(stud_train_ld, train_ep)

        print('Train student: Epoch %d, Step %d' % (ep, i))
        tld = eval_student(student, test_dl)
        stud_val_ld = update_lossdict(stud_val_ld, tld)

        train_ep = eval_student(student, train_dl)
        stud_train_ld = update_lossdict(stud_train_ld, train_ep)

    return stud_train_ld, stud_val_ld, student


# Evaluate student on complete train/test set.
def eval_student(student, dl):
    student.eval()
    net_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dl:
            data, target = data.to(device), target.to(device)
            output = student(data)
            net_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    net_loss /= len(dl.dataset)
    
    acc = 100. * correct / len(dl.dataset) 

    print('\n Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        net_loss, correct, len(dl.dataset),
        100. * correct / len(dl.dataset)))
    return {'epoch_loss': net_loss, 'epoch_acc': acc}


savefol = 'weightlearn-{dataset}-{seed}-{innerlr}-{arch}-{studentarch}-{innersteps}'.format(dataset=args.dataset,
                         seed=args.seed,
                         innerlr=args.inner_lr,
                         outerlr=args.outer_lr,
                         arch=args.arch,
                         studentarch=args.studentarch,
                         innersteps=args.inner_steps)

if args.teachckpt:
    tsd = torch.load(args.teachckpt)
else:
    tsd = None

if args.baseline:
    res_baseline_train, res_baseline_val, stud_baseline = baseline(train_loader, test_loader)
    resfile = f'{args.studentarch}-seed{args.seed}_results_baseline_longtrain.ckpt'
    with open(os.path.join(teacher_savefol, resfile), 'wb') as f:
        saveres = [res_baseline_train, res_baseline_val]
        pickle.dump(saveres, f)

else:
    res_tsn_train, res_tsn_val, stud_tsn = baseline(train_loader, test_loader, teach_state_dict=tsd)
    resfile = f'cbr2-{args.studentarch}-seed{args.seed}_results_tsn_longtrain.ckpt'
    with open(os.path.join(teacher_savefol, resfile), 'wb') as f:
        saveres = [res_tsn_train, res_tsn_val]
        pickle.dump(saveres, f)





