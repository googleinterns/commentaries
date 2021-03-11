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


import copy
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
parser.add_argument('--dataset', help='dataset: mnist or cifar10 or cifar100', type=str, default='cifar10')
parser.add_argument('--seed', help='seed', type=int, default=0)
parser.add_argument('--inner-lr', help='LR for inner optim (of student)', type=float, default=1e-4)
parser.add_argument('--outer-lr', help='LR for outer optim (of teacher)', type=float, default=1e-4)
parser.add_argument('--gpu', help='gpu to use', type=int, default=0)
parser.add_argument('--studentarch', help='student architecture', type=str, default='cbr')
parser.add_argument('--arch', help='teacher architecture; curr uses curriculum (itr of training), nocurr does not', type=str, default='curr')
parser.add_argument('--inner-steps', help='num inner steps', type=int, default=100)
parser.add_argument('--train-steps', help='num train steps', type=int, default=100)
args = parser.parse_args()

torch.manual_seed(args.seed)


train_loader, val_loader, test_loader = get_data(args.dataset)

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


# Train the student network for a specified number of inner loop optimisation steps
def do_train_student(student, optimizer, teacher, train_dl, inner_steps = args.inner_steps):
    lossdict = {
        'stud_y_loss' : [],
        'stud_loss' : [],
        'stud_acc' : []
    }
    student.train()
    teacher.train()
    for i, (x,y) in enumerate(train_dl):
        x = x.to(device)
        y = y.to(device)
        y_loss, acc  = get_loss(student, teacher, x, y, i)

        loss = y_loss
        
        # higher library exposes this interface for zero_grad on optimizer, loss.backward(), and optim.step()
        optimizer.step(loss)
    
        # logging
        lossdict['stud_y_loss'].append(y_loss.item())
        lossdict['stud_loss'].append(loss.item())
        lossdict['stud_acc'].append(acc)

        if i % 100 == 0:
            print('Train student: Step ', i)
        if i == inner_steps-1:
            break
    return lossdict, student



def do_train_teacher(teacher, optimizer, student, train_dl):
    lossdict = {
        'teach_loss' : [],
    }

    student.train()
    teacher.train()
    netloss = None
    for i, (x,y) in enumerate(train_dl):
        x = x.to(device)
        y = y.to(device)
        # Set val=True to get unweighted loss
        y_loss_stud, acc_stud = get_loss(student, teacher, x, y, i, val=True)

        # Teacher is trained to minimise the student's classification error loss after training
        if netloss is None:
            netloss = y_loss_stud
        else:
            netloss += y_loss_stud 

        # Logging
        lossdict['teach_loss'].append(y_loss_stud.item())

        if i % 500 == 0:
            print('Train teacher: Step ', i)

    # Accumulate losses and do the step after we process outer_steps number of batches.
    netloss = netloss/len(train_dl)
    netloss.backward(retain_graph=True)

    return lossdict, teacher

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



def train(train_dl, val_dl, test_dl, num_steps = args.train_steps, tsd=None, ssd=None):

    if tsd is None:
        if args.arch =='curr':
            teacher = ConvNetTeacher(args.dataset, args.inner_steps).to(device)
        elif args.arch == 'nocurr':
            teacher = ConvNetTeacher2(args.dataset, args.inner_steps).to(device)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    teacher_optim = torch.optim.Adam(teacher.parameters(), lr=args.outer_lr)

    stud_state_dict = None if ssd is None else copy.deepcopy(ssd)
    stud_train_ld = {}
    stud_val_ld = {}
    stud_test_ld = {}

    teacher_train_ld = {}

    for ep in range(num_steps):
        if args.studentarch =='cbr':
            student = ConvNetStudent(args.dataset).to(device)
        else:
            # only support small convnet for student during commentary learning (memory constraints)
            raise NotImplementedError

        # This helps keep the init params fixed for each student
        if stud_state_dict is None:
            stud_state_dict = copy.deepcopy(student.state_dict())
        else:
            student.load_state_dict(stud_state_dict, strict=False)

        stud_optim = torch.optim.Adam(student.parameters(), lr=args.inner_lr)

        teacher_optim.zero_grad()
        
        # The outer loop optimiser (teacher optimiser) does not change the student params, so copy_initial_weights can be True or False
        with higher.innerloop_ctx(student, stud_optim, copy_initial_weights=False) as (fnet, diffopt):
            student_ld, fnet = do_train_student(fnet, diffopt, teacher, train_dl)
            # Note: do_train_student does not edit the .grad field of teacher parameters
            # Therefore, no need to clear teacher grads before do_train_teacher.
            teacher_ld, teacher = do_train_teacher(teacher, teacher_optim, fnet, val_dl)
            student.load_state_dict(fnet.state_dict())

        teacher_optim.step()

        stud_train_ld = update_lossdict(stud_train_ld, student_ld)
        teacher_train_ld = update_lossdict(teacher_train_ld, teacher_ld)

        print("Teacher training step: ", ep)

        tld = eval_student(student, test_dl)
        stud_test_ld = update_lossdict(stud_test_ld, tld)
        vld = eval_student(student, val_dl)
        stud_val_ld = update_lossdict(stud_val_ld, vld)
        
        train_ep = eval_student(student, train_dl)
        stud_train_ld = update_lossdict(stud_train_ld, train_ep) 
        
        if ep % 10 == 0:
            teach_path = os.path.join(savefol, 'teacher-{arch}-step{sp}.ckpt'.format(arch=args.arch, sp=ep))
            torch.save(teacher.state_dict(), teach_path)
        
    return teacher, stud_train_ld, teacher_train_ld, stud_val_ld, stud_test_ld, stud_state_dict



savefol = 'weightlearn-{dataset}-{seed}-{innerlr}-{outerlr}-{arch}-{studentarch}-{innersteps}-nsteps{nsteps}'.format(dataset=args.dataset,
                         seed=args.seed,
                         innerlr=args.inner_lr,
                         outerlr=args.outer_lr,
                         arch=args.arch,
                         studentarch=args.studentarch,
                         innersteps=args.inner_steps,
                         nsteps=args.train_steps)

os.makedirs(savefol, exist_ok=True)

student_sd = None

res = train(train_loader, val_loader, test_loader, ssd=student_sd, tsd=None)

teacher, stud_train_ld, teach_train_ld, stud_val_ld, stud_test_ld, stud_sd = res

saveres = [stud_train_ld, teach_train_ld, stud_val_ld, stud_test_ld]

with open(os.path.join(savefol, 'results.pkl'), 'wb') as f:
    pickle.dump(saveres, f)

# save student for analysis later
stud_path = os.path.join(savefol, 'stud_sd.ckpt')
torch.save(stud_sd, stud_path)

# save teacher for analysis later
teach_path = os.path.join(savefol, 'teacher.ckpt')
torch.save(teacher.state_dict(), teach_path)




